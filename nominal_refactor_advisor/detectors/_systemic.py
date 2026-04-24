"""Systemic detector implementations.

This module holds the earlier detector families that focus on orchestration,
axis authority, registration, and other repo-wide architectural smells.
"""

from __future__ import annotations

from ._base import *
from ._helpers import *

class RepeatedPrivateMethodDetector(FiberCollectedShapeIssueDetector):
    detector_id = "repeated_private_methods"
    observation_kind = ObservationKind.METHOD_SHAPE
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Repeated non-orthogonal method skeleton across classes",
        why=(
            "Shared orchestration logic is duplicated across a behavior family. The docs say this shared "
            "non-orthogonal logic should move into an ABC with a concrete template method, leaving only "
            "orthogonal hooks in subclasses."
        ),
        capability_gap="single authoritative algorithm for a nominal behavior family",
        relation_context="same method role across sibling classes",
        confidence=HIGH_CONFIDENCE,
        certification=CERTIFIED,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        observation_tags=(
            ObservationTag.NORMALIZED_AST,
            ObservationTag.CLASS_FAMILY,
            ObservationTag.METHOD_ROLE,
        ),
    )

    def _module_shapes(self, module: ParsedModule) -> tuple[object, ...]:
        return tuple(
            _collect_typed_family_items(module, MethodShapeFamily, MethodShape)
        )

    def _include_shape(self, shape: object, config: DetectorConfig) -> bool:
        method = _as_method_shape(shape)
        return bool(
            method.class_name
            and method.statement_count >= config.min_duplicate_statements
        )

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
        return self.finding_spec.build(
            self.detector_id,
            (
                f"{len(methods)} methods across {len(class_names)} classes share the same normalized AST shape."
            ),
            evidence,
            relation_context=relation,
            scaffold=_abc_scaffold_for_methods(methods),
            codemod_patch=_abc_patch_for_methods(methods),
            metrics=RepeatedMethodMetrics.from_duplicate_family(
                duplicate_site_count=len(methods),
                statement_count=methods[0].statement_count,
                class_count=len(class_names),
                method_symbols=tuple(method.symbol for method in methods),
                shared_statement_texts=methods[0].statement_texts,
            ),
        )


class InheritanceHierarchyCandidateDetector(IssueDetector):
    detector_id = "inheritance_hierarchy_candidate"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Classes cluster into an ABC hierarchy candidate",
        why=(
            "The same set of classes repeats multiple non-orthogonal method skeletons. The docs say this is a "
            "strong signal that the family should be factored into an ABC with one concrete template method "
            "layer; orthogonal reusable concerns can then live in mixins so MRO preserves declared precedence."
        ),
        capability_gap="single authoritative inheritance hierarchy for a duplicated behavior family",
        relation_context="same class set repeats several method roles across the same family boundary",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        observation_tags=(
            ObservationTag.REPEATED_METHOD_ROLES,
            ObservationTag.CLASS_FAMILY,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        repeated_methods = tuple(
            method
            for module in modules
            for method in _collect_typed_family_items(
                module, MethodShapeFamily, MethodShape
            )
            if method.class_name
            and method.statement_count >= config.min_duplicate_statements
        )
        graph = ObservationGraph(
            tuple(method.structural_observation for method in repeated_methods)
        )
        lookup = _carrier_lookup(tuple(repeated_methods))

        findings: list[RefactorFinding] = []
        for cohort in graph.coherence_cohorts_for(
            ObservationKind.METHOD_SHAPE,
            StructuralExecutionLevel.FUNCTION_BODY,
            minimum_witnesses=2,
            minimum_fibers=1,
        ):
            groups = [
                tuple(
                    _as_method_shape(item)
                    for item in _materialize_observations(fiber.observations, lookup)
                )
                for fiber in cohort.fibers
            ]
            if not groups:
                continue
            class_names = frozenset(cohort.nominal_witnesses)
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
            evidence = [
                SourceLocation(method.file_path, method.lineno, method.symbol)
                for methods in groups
                for method in methods
            ]
            findings.append(
                self.finding_spec.build(
                    self.detector_id,
                    (
                        f"Classes {', '.join(sorted(class_names))} share {len(groups)} repeated method-shape groups and repeated method roles that likely want one ABC family."
                    ),
                    tuple(evidence[:8]),
                    scaffold=_abc_family_scaffold(class_names, groups),
                    codemod_patch=_abc_family_patch(class_names, groups),
                    metrics=HierarchyCandidateMetrics(
                        duplicate_group_count=len(groups),
                        class_count=len(class_names),
                    ),
                )
            )
        return findings


class OrchestrationHubDetector(CandidateFindingDetector):
    detector_id = "orchestration_hub"
    finding_spec = FindingSpec(
        pattern_id=PatternId.STAGED_ORCHESTRATION,
        title="Oversized orchestration hub",
        why=(
            "One function is owning too many control branches, helper calls, and phase transitions at once. "
            "The architecture wants explicit staged boundaries so the orchestration surface remains nominal and legible."
        ),
        capability_gap="explicit staged orchestration boundaries with named phase contracts",
        relation_context="one owner centralizes many operational phases and helper families",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return tuple(
            profile
            for profile in _function_profiles(module)
            if profile.line_count >= config.min_orchestration_function_lines
            and profile.branch_count >= config.min_orchestration_branches
            and profile.call_count >= config.min_orchestration_calls
        )

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        profile = cast(FunctionProfile, candidate)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{profile.qualname}` concentrates {profile.line_count} lines, {profile.branch_count} branches, and {profile.call_count} calls across {profile.callee_family_count} callee families in one owner."
            ),
            (profile.evidence,),
            scaffold=_orchestration_stage_scaffold(profile),
            codemod_patch=_orchestration_stage_patch(profile),
            metrics=OrchestrationMetrics(
                function_line_count=profile.line_count,
                branch_site_count=profile.branch_count,
                call_site_count=profile.call_count,
                parameter_count=len(profile.parameter_names),
                callee_family_count=profile.callee_family_count,
            ),
        )


class PrivateCohortShouldBeModuleDetector(CandidateFindingDetector):
    detector_id = "private_cohort_should_be_module"
    finding_spec = FindingSpec(
        pattern_id=PatternId.STAGED_ORCHESTRATION,
        title="Private subsystem cohort wants its own module",
        why=(
            "One module is carrying a tightly-coupled private subsystem cohort as if it were a whole package. "
            "The architecture wants a dedicated module for that bounded context, with the original file reduced to orchestration or public entry points."
        ),
        capability_gap="explicit module-level subsystem boundaries with extracted private cohorts",
        relation_context="one file contains a dense private context/result/helper family that should move together",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _private_cohort_should_be_module_candidates(module, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        cohort = cast(PrivateCohortShouldBeModuleCandidate, candidate)
        shared_tokens = ", ".join(cohort.shared_tokens[:3]) or "subsystem"
        sample_symbols = ", ".join(
            symbol.symbol
            for symbol in sorted(
                cohort.symbols,
                key=lambda item: (-item.line_count, item.line, item.symbol),
            )[:3]
        )
        target_module = _suggest_private_cohort_module_name(cohort)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{cohort.module_name}` carries a private {shared_tokens} cohort across "
                f"{len(cohort.symbols)} top-level symbols / {cohort.total_cohort_lines} lines "
                f"inside a {cohort.module_line_count}-line module; extract `{sample_symbols}` "
                f"into a dedicated `{target_module}.py` module."
            ),
            cohort.evidence,
            scaffold=(
                f"# {target_module}.py\n"
                "@dataclass(frozen=True)\n"
                f"class {_camel_case('_'.join(cohort.shared_tokens[:2]) or 'subsystem')}Context:\n"
                "    ...\n\n"
                f"def run_{'_'.join(cohort.shared_tokens[:2]) or 'subsystem'}(...):\n"
                "    ...\n\n"
                "# Move the private context/result carriers and worker helpers here.\n"
                "# Leave only public orchestration entry points in the original module."
            ),
            codemod_patch=(
                f"# Extract the private {shared_tokens} cohort into `{target_module}.py`.\n"
                "# Move the cohort's private dataclasses, helper functions, and result carriers together.\n"
                "# Import the extracted helpers back into the original module only where public entry points still need them.\n"
                "# Keep sequencing, public APIs, and thin phase boundaries in the original file."
            ),
        )


class ParameterThreadFamilyDetector(CandidateFindingDetector):
    detector_id = "parameter_thread_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_CONTEXT,
        title="Repeated threaded semantic parameter family",
        why=(
            "Several helpers keep re-threading the same semantic parameter bundle instead of carrying one nominal context. "
            "That weakens provenance and makes each helper signature a partially duplicated view of the same authority."
        ),
        capability_gap="one authoritative context/request record for a shared semantic parameter family",
        relation_context="the same semantic parameter bundle is threaded through several sibling helpers",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _parameter_thread_family_candidates(module, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        parameter_family = cast(ParameterThreadFamilyCandidate, candidate)
        function_names = tuple(item.qualname for item in parameter_family.functions)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Functions {', '.join(function_names[:4])} thread the same semantic parameter family `{', '.join(parameter_family.shared_parameter_names)}` across {len(parameter_family.functions)} helpers."
            ),
            tuple(item.evidence for item in parameter_family.functions[:6]),
            scaffold=_authoritative_context_scaffold(parameter_family),
            codemod_patch=_authoritative_context_patch(parameter_family),
            metrics=ParameterThreadMetrics(
                function_count=len(parameter_family.functions),
                shared_parameter_count=len(parameter_family.shared_parameter_names),
                shared_parameter_names=parameter_family.shared_parameter_names,
            ),
        )


class EnumStrategyDispatchDetector(CandidateFindingDetector):
    detector_id = "enum_strategy_dispatch"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_STRATEGY_FAMILY,
        title="Enum strategy ladder wants nominal family",
        why=(
            "A closed enum/member dispatch ladder is choosing among behavior implementations inline. "
            "That wants an ABC-backed strategy family so each implementation guarantees one common method and the caller stops branching."
        ),
        capability_gap="nominal strategy family with one guaranteed call surface",
        relation_context="one owner branches over a closed enum/member family instead of delegating to implementation classes",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _enum_strategy_dispatch_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        dispatch_candidate = cast(EnumStrategyDispatchCandidate, candidate)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{dispatch_candidate.qualname}` branches on `{dispatch_candidate.dispatch_axis}` across closed cases {', '.join(dispatch_candidate.case_names)} and should delegate to a nominal strategy family."
            ),
            (dispatch_candidate.evidence,),
            scaffold=_nominal_strategy_scaffold(dispatch_candidate),
            codemod_patch=_nominal_strategy_patch(dispatch_candidate),
            metrics=DispatchCountMetrics(
                dispatch_site_count=len(dispatch_candidate.case_names),
                dispatch_axis=dispatch_candidate.dispatch_axis,
                literal_cases=dispatch_candidate.case_names,
            ),
        )


class RepeatedEnumStrategyDispatchDetector(CandidateFindingDetector):
    detector_id = "repeated_enum_strategy_dispatch"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_STRATEGY_FAMILY,
        title="Repeated closed-strategy dispatch should centralize in one nominal strategy family",
        why=(
            "Several owners re-dispatch the same closed enum family inline. The docs treat that as duplicated "
            "strategy orchestration: dispatch should happen once through one authoritative nominal strategy family "
            "or one shared strategy substrate."
        ),
        capability_gap="single authoritative nominal strategy family for a repeated closed dispatch axis",
        relation_context="same closed enum family is re-dispatched across sibling functions or methods",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _repeated_enum_strategy_dispatch_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        dispatch_candidate = cast(RepeatedEnumStrategyDispatchCandidate, candidate)
        evidence = tuple(
            item.evidence for item in dispatch_candidate.functions[:6]
        )
        representative = dispatch_candidate.functions[0]
        function_names = ", ".join(
            item.qualname for item in dispatch_candidate.functions[:4]
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Functions {function_names} each re-dispatch `{dispatch_candidate.enum_family}` cases "
                f"{', '.join(dispatch_candidate.shared_case_names)} inline."
            ),
            evidence,
            scaffold=_nominal_strategy_scaffold(representative),
            codemod_patch=_nominal_strategy_patch(representative),
            metrics=DispatchCountMetrics(
                dispatch_site_count=len(dispatch_candidate.functions),
                dispatch_axis=dispatch_candidate.enum_family,
                literal_cases=dispatch_candidate.shared_case_names,
            ),
        )


class SplitDispatchAuthorityDetector(CandidateFindingDetector):
    detector_id = "split_dispatch_authority"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_STRATEGY_FAMILY,
        title="Cooperating dispatch layers should collapse into one product-family authority",
        why=(
            "The docs treat repeated cooperating dispatch layers as split authority. When one orchestration function "
            "selects a strategy-family implementation and separately routes another axis through `singledispatch`, "
            "the operation usually wants one authoritative product-family policy or one request-dispatched plan."
        ),
        capability_gap="single authoritative product-family or request-dispatched policy for cooperating dispatch axes",
        relation_context="one orchestrator combines a strategy-family selector with a separate singledispatch generic",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.FACTORY_DISPATCH,
            ObservationTag.REPEATED_METHOD_ROLES,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _split_dispatch_authority_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        dispatch_candidate = cast(SplitDispatchAuthorityCandidate, candidate)
        evidence = (
            dispatch_candidate.evidence,
            SourceLocation(
                dispatch_candidate.file_path,
                dispatch_candidate.selector_line,
                f"{dispatch_candidate.strategy_root_name}.{dispatch_candidate.selector_method_name}",
            ),
            SourceLocation(
                dispatch_candidate.file_path,
                dispatch_candidate.generic_line,
                dispatch_candidate.generic_function_name,
            ),
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{dispatch_candidate.qualname}` combines strategy selector "
                f"`{dispatch_candidate.strategy_root_name}.{dispatch_candidate.selector_method_name}({dispatch_candidate.strategy_axis_expression})` "
                f"with singledispatch `{dispatch_candidate.generic_function_name}({dispatch_candidate.generic_axis_expression})` "
                f"through callback `{dispatch_candidate.bridge_callback_name}`, splitting one operation across two dispatch authorities."
            ),
            evidence,
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class DispatchPlan:\n"
                "    strategy: object\n"
                "    source_type: type[object]\n\n"
                "class ProductPolicy(ABC):\n"
                "    plan_key: ClassVar[DispatchPlan]\n"
                "    def run(self, request): ...\n"
            ),
            codemod_patch=(
                f"# Collapse `{dispatch_candidate.strategy_root_name}` and `{dispatch_candidate.generic_function_name}` under one product-family authority.\n"
                "# Let one nominal plan/policy own both `{dispatch_candidate.strategy_axis_expression}` and `{dispatch_candidate.generic_axis_expression}` so the orchestrator dispatches once."
            ),
            metrics=DispatchCountMetrics(
                dispatch_site_count=2,
                dispatch_axis=(
                    f"{dispatch_candidate.strategy_axis_expression} x "
                    f"{dispatch_candidate.generic_axis_expression}"
                ),
                literal_cases=(
                    *dispatch_candidate.strategy_case_names[:3],
                    *dispatch_candidate.generic_case_names[:3],
                ),
            ),
        )


class EmptyLeafProductFamilyDetector(CandidateFindingDetector):
    detector_id = "empty_leaf_product_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.CLOSED_FAMILY_DISPATCH,
        title="Empty multiple-inheritance leaves should collapse into one product-family authority",
        why=(
            "The docs allow mixins for orthogonal reusable concerns, but empty leaf classes that merely enumerate "
            "all combinations of two reusable axes are usually a handwritten product table in inheritance form. "
            "That product should become one keyed authority or one product-family selector."
        ),
        capability_gap="single authoritative keyed product family instead of empty inheritance combinations",
        relation_context="empty leaf classes encode the full Cartesian product of two reusable inheritance axes",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.REPEATED_METHOD_ROLES,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _empty_leaf_product_family_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        product_candidate = cast(EmptyLeafProductFamilyCandidate, candidate)
        left_axis = ", ".join(product_candidate.left_axis_base_names)
        right_axis = ", ".join(product_candidate.right_axis_base_names)
        leaf_preview = ", ".join(product_candidate.leaf_class_names[:6])
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Empty leaf classes {leaf_preview} encode `{left_axis}` x `{right_axis}` through multiple inheritance instead of one product-family authority."
            ),
            product_candidate.evidence,
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class ProductRule:\n"
                "    axis_left: object\n"
                "    axis_right: object\n"
                "    policy_type: type[object]\n\n"
                "PRODUCT_RULES = (...)\n"
            ),
            codemod_patch=(
                "# Replace the empty Cartesian-product leaf classes with one keyed product table or one nominal selector family.\n"
                "# Keep only irreducible axis-local behavior on the reusable bases; do not encode the cross product as `pass` subclasses."
            ),
            metrics=DispatchCountMetrics(
                dispatch_site_count=len(product_candidate.leaf_class_names),
                dispatch_axis=(
                    f"{' | '.join(product_candidate.left_axis_base_names)} x "
                    f"{' | '.join(product_candidate.right_axis_base_names)}"
                ),
                literal_cases=product_candidate.leaf_class_names,
            ),
        )


class ClosedConstantSelectorDetector(CandidateFindingDetector):
    detector_id = "closed_constant_selector"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Closed selector over sibling constants should derive from one selector table",
        why=(
            "The docs treat branch ladders that choose among sibling specs, plans, contracts, or other immutable "
            "constants as duplicated selector logic once the constant family already exists. The selector should "
            "collapse into one authoritative keyed table or selector record so wrappers and downstream views are derived."
        ),
        capability_gap="single authoritative selector table for a closed constant family",
        relation_context="one function branches over a small predicate family and returns sibling constants or one shared wrapper around them",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.PREDICATE_CHAIN,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _closed_constant_selector_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        selector_candidate = cast(ClosedConstantSelectorCandidate, candidate)
        constants_preview = ", ".join(selector_candidate.constant_names[:4])
        guard_preview = ", ".join(selector_candidate.guard_expressions[:2])
        family_label = (
            selector_candidate.common_constructor_name
            or selector_candidate.family_suffix
            or "selected constant family"
        )
        wrapper_summary = (
            f"`{selector_candidate.wrapper_name}(...)` around "
            if selector_candidate.wrapper_name is not None
            else ""
        )
        guard_summary = (
            f"guards `{guard_preview}` and default fallback"
            if selector_candidate.guard_expressions
            else "a closed fallback ladder"
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{selector_candidate.qualname}` branches over {guard_summary}, returning {wrapper_summary}"
                f"sibling constants {constants_preview} from `{family_label}`."
            ),
            selector_candidate.evidence,
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class SelectorRule:\n"
                "    key: object\n"
                "    selected: object\n\n"
                "SELECTOR_RULES = (\n"
                "    SelectorRule(key=..., selected=...),\n"
                ")\n"
                "_SELECTED_BY_KEY = {rule.key: rule.selected for rule in SELECTOR_RULES}\n"
            ),
            codemod_patch=(
                f"# Replace manual branches in `{selector_candidate.qualname}` with one authoritative selector table.\n"
                "# Select the sibling constant once, then apply any shared wrapper outside the selector."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(selector_candidate.constant_names),
                field_count=max(len(selector_candidate.guard_expressions), 1),
                mapping_name=selector_candidate.wrapper_name or family_label,
                field_names=selector_candidate.constant_names,
                source_name=selector_candidate.qualname,
            ),
        )


class DerivedWrapperSpecShadowDetector(CandidateFindingDetector):
    detector_id = "derived_wrapper_spec_shadow"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Generated wrapper spec family should collapse into the authoritative spec family",
        why=(
            "The docs treat writable wrapper-spec tables as secondary authorities when they just point back at an "
            "existing spec family and feed code generation. Wrapper metadata should live on the authoritative spec "
            "records so generated wrappers are derived from one source rather than synchronized across parallel tables."
        ),
        capability_gap="single authoritative spec family carrying wrapper-generation metadata",
        relation_context="secondary spec table references an authoritative spec family entry-by-entry and is only consumed by wrapper generation",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        observation_tags=(
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.SCOPED_SHAPE_WRAPPER,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _derived_wrapper_spec_shadow_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        shadow_candidate = cast(DerivedWrapperSpecShadowCandidate, candidate)
        primary_family_label = (
            shadow_candidate.primary_family_name or shadow_candidate.primary_constructor_name
        )
        constant_preview = ", ".join(shadow_candidate.primary_constant_names[:4])
        builder_preview = ", ".join(shadow_candidate.builder_names[:3])
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{shadow_candidate.derived_family_name}` re-encodes wrapper metadata over authoritative family "
                f"`{primary_family_label}` through link field `{shadow_candidate.link_field_name}` for {constant_preview}, "
                f"then feeds generated wrappers via {builder_preview}."
            ),
            shadow_candidate.evidence,
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class ExecutionSpec:\n"
                "    key: object\n"
                "    runner: object\n"
                "    wrapper_name: str | None = None\n"
                "    wrapper_defaults: dict[str, object] = field(default_factory=dict)\n\n"
                "def build_wrapper(spec: ExecutionSpec): ...\n"
            ),
            codemod_patch=(
                f"# Remove parallel family `{shadow_candidate.derived_family_name}`.\n"
                f"# Move `{', '.join(shadow_candidate.extra_field_names) or 'wrapper metadata'}` onto the authoritative "
                f"`{shadow_candidate.primary_constructor_name}` records and derive wrappers directly from that family."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(shadow_candidate.primary_constant_names),
                field_count=max(len(shadow_candidate.extra_field_names), 1),
                mapping_name=shadow_candidate.derived_family_name,
                field_names=shadow_candidate.extra_field_names,
                source_name=primary_family_label,
                identity_field_names=(shadow_candidate.link_field_name,),
            ),
        )


class ModuleKeyedSelectionHelperDetector(CandidateFindingDetector):
    detector_id = "module_keyed_selection_helper"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Local keyed-selection helper should collapse into the generic keyed-record table",
        why=(
            "The docs push reusable table/index machinery into one authoritative substrate. When a module defines a "
            "local selection-rule dataclass, a dict-index builder, and a keyed lookup helper that power multiple rule "
            "tables, it is reintroducing a second keyed-table framework instead of reusing the generic keyed-record helper."
        ),
        capability_gap="single authoritative keyed-record table substrate reused across module-level selector tables",
        relation_context="module-local selection helper framework powers multiple keyed rule tables",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.CLASS_FAMILY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _module_keyed_selection_helper_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        helper_candidate = cast(ModuleKeyedSelectionHelperCandidate, candidate)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{helper_candidate.rule_class_name}`, `{helper_candidate.helper_function_name}`, and "
                f"`{helper_candidate.lookup_function_name}` implement a local keyed-selection substrate for "
                f"{', '.join(helper_candidate.rule_table_names[:4])} and indexes {', '.join(helper_candidate.index_table_names[:4])}."
            ),
            helper_candidate.evidence,
            scaffold=(
                "KeyT = TypeVar(\"KeyT\")\n"
                "RecordT = TypeVar(\"RecordT\")\n\n"
                "@dataclass(frozen=True)\n"
                "class KeyedRecordTable(Generic[KeyT, RecordT]):\n"
                "    records: tuple[RecordT, ...]\n"
                "    key_of: Callable[[RecordT], KeyT]\n"
                "    def require(self, key: KeyT, *, missing_error=None) -> RecordT: ...\n"
            ),
            codemod_patch=(
                f"# Remove local keyed-selection helper `{helper_candidate.rule_class_name}` / "
                f"`{helper_candidate.helper_function_name}` / `{helper_candidate.lookup_function_name}`.\n"
                "# Re-express these rule tables through the shared KeyedRecordTable substrate."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(helper_candidate.rule_table_names),
                field_count=1,
                mapping_name=helper_candidate.rule_class_name,
                field_names=(helper_candidate.selected_field_name,),
                source_name=helper_candidate.helper_function_name,
                identity_field_names=("key",),
            ),
        )


class CrossModuleAxisShadowFamilyDetector(CrossModuleCandidateDetector):
    detector_id = "cross_module_axis_shadow_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_STRATEGY_FAMILY,
        title="Cross-module shadow family should collapse into one axis authority",
        why=(
            "The docs require one authoritative owner per closed semantic axis. When one module already owns an enum/keyed "
            "family nominally and another module reintroduces a second family over the same cases, the axis has split "
            "authority and local behavior should derive from the authoritative family instead."
        ),
        capability_gap="single authoritative closed-axis family reused across modules",
        relation_context="same keyed enum axis is modeled by an authoritative family in one module and a shadow selector family in another",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.FACTORY_DISPATCH,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _cross_module_axis_shadow_family_candidates(modules)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        shadow_candidate = cast(CrossModuleAxisShadowFamilyCandidate, candidate)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Axis `{shadow_candidate.key_type_name}` is already owned by "
                f"`{shadow_candidate.authoritative.family_name}` but re-encoded by "
                f"`{shadow_candidate.shadow.family_name}.{shadow_candidate.selector_method_name}` "
                f"across cases {', '.join(shadow_candidate.shared_case_names[:4])}."
            ),
            shadow_candidate.evidence,
            scaffold=(
                _axis_policy_registry_scaffold("invariant(self)")
                + "\n\n"
                f"def run_with_axis(axis: {_AXIS_POLICY_KEY_TYPE_NAME}, ...):\n"
                f"    policy = {_AXIS_POLICY_ROOT_NAME}.for_key(axis)\n"
                "    # derive local execution from authoritative policy facts\n"
            ),
            codemod_patch=(
                f"# Remove shadow family `{shadow_candidate.shadow.family_name}`.\n"
                f"# Derive local behavior from authoritative family `{shadow_candidate.authoritative.family_name}` instead of re-owning axis `{shadow_candidate.key_type_name}`."
            ),
            metrics=_axis_dispatch_metrics(
                shadow_candidate.shared_case_names,
                shadow_candidate.key_type_name,
            ),
        )


class ResidualClosedAxisBranchingDetector(CrossModuleCandidateDetector):
    detector_id = "residual_closed_axis_branching"
    finding_spec = FindingSpec(
        pattern_id=PatternId.CLOSED_FAMILY_DISPATCH,
        title="Manual closed-axis branching should derive from existing keyed authority",
        why=(
            "The docs require one authoritative owner per closed enum/key axis. When a keyed nominal family already "
            "owns that axis, downstream `if`/`match` ladders over the same cases become residual shadow dispatch."
        ),
        capability_gap="behavior derived from authoritative keyed family rather than downstream enum branching",
        relation_context="function branches on an enum axis already owned by a keyed nominal family in another module",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.BRANCH_DISPATCH,
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _residual_closed_axis_branching_candidates(modules)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        residual_candidate = cast(ResidualClosedAxisBranchingCandidate, candidate)
        authoritative_family_names = ", ".join(
            family_name
            for family_name, _, _ in residual_candidate.authoritative_families[:4]
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{residual_candidate.qualname}` branches {residual_candidate.branch_site_count} time(s) on axis "
                f"`{residual_candidate.key_type_name}` across cases {', '.join(residual_candidate.case_names)}, "
                f"even though authoritative family `{authoritative_family_names}` already owns that axis."
            ),
            residual_candidate.evidence,
            scaffold=(
                _axis_policy_registry_scaffold("apply(self, context)")
                + "\n\n"
                "def run(context):\n"
                f"    policy = {_AXIS_POLICY_ROOT_NAME}.for_key(context.axis)\n"
                "    return policy.apply(context)\n"
            ),
            codemod_patch=(
                f"# Remove residual `{residual_candidate.key_type_name}` branch ladder in `{residual_candidate.qualname}`.\n"
                "# Delegate through the existing keyed family authority and keep only case-local residue on the policy leaves."
            ),
            metrics=DispatchCountMetrics(
                dispatch_site_count=residual_candidate.branch_site_count,
                dispatch_axis=residual_candidate.key_type_name,
                literal_cases=residual_candidate.case_names,
            ),
        )


class ParallelKeyedAxisFamilyDetector(CrossModuleCandidateDetector):
    detector_id = "parallel_keyed_axis_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_STRATEGY_FAMILY,
        title="Parallel keyed families should collapse into one axis authority",
        why=(
            "The docs require one authoritative nominal owner per closed semantic axis. When two modules each define a "
            "keyed family over the same enum/key cases, the axis has split ownership even if both sides are nominal."
        ),
        capability_gap="single cross-module keyed-axis authority with module-local adapters derived from it",
        relation_context="same keyed enum axis is modeled by multiple nominal families across modules",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.FACTORY_DISPATCH,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _parallel_keyed_axis_family_candidates(modules)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        family_candidate = cast(ParallelKeyedAxisFamilyCandidate, candidate)
        shared_cases = ", ".join(family_candidate.shared_case_names[:4])
        label_clause = ""
        if (
            family_candidate.left.family_label is not None
            and family_candidate.left.family_label == family_candidate.right.family_label
        ):
            label_clause = (
                f" Both declare family label `{family_candidate.left.family_label}`."
            )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Axis `{family_candidate.key_type_name}` is owned in parallel by "
                f"`{family_candidate.left.family_name}` and `{family_candidate.right.family_name}` "
                f"across cases {shared_cases}.{label_clause}"
            ),
            family_candidate.evidence,
            scaffold=(
                _axis_policy_registry_scaffold(
                    "invariant(self)",
                    "runtime_adapter(self, context)",
                )
                + "\n\n"
                "# Keep one authoritative keyed family and let secondary modules derive local adapters/specs from it."
            ),
            codemod_patch=(
                f"# Collapse `{family_candidate.left.family_name}` and `{family_candidate.right.family_name}` onto one authoritative keyed family.\n"
                "# Move the irreducible case-specific hooks to that family or to a single derived adapter table, not two parallel nominal roots."
            ),
            metrics=_axis_dispatch_metrics(
                family_candidate.shared_case_names,
                family_candidate.key_type_name,
            ),
        )


class ParallelKeyedTableAxisDetector(CrossModuleCandidateDetector):
    detector_id = "parallel_keyed_table_axis"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Parallel enum-keyed tables across modules should collapse into one axis record",
        why=(
            "The docs require one authoritative writable owner per closed semantic axis. "
            "When multiple modules maintain separate enum-keyed tables over the same cases, the axis is split across parallel metadata maps."
        ),
        capability_gap="single authoritative enum-keyed row family with derived module-local projections",
        relation_context="same closed enum/key axis is encoded by multiple keyed tables across modules",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.PROJECTION_DICT,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.BUILDER_CALL,
        ),
    )

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _parallel_keyed_table_axis_candidates(modules)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        table_candidate = cast(ParallelKeyedTableAxisCandidate, candidate)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Axis `{table_candidate.key_type_name}` is restated by `{table_candidate.left.table_name}` and "
                f"`{table_candidate.right.table_name}` across cases {', '.join(table_candidate.shared_case_names[:4])}."
            ),
            table_candidate.evidence,
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class AxisRow:\n"
                "    key: AxisEnum\n"
                "    primary: object\n"
                "    secondary: object | None = None\n\n"
                "AXIS_ROWS = (\n"
                "    AxisRow(key=AxisEnum.ALPHA, primary=..., secondary=...),\n"
                ")\n"
                "AXIS_ROW_BY_KEY = {row.key: row for row in AXIS_ROWS}\n"
            ),
            codemod_patch=(
                f"# Collapse `{table_candidate.left.table_name}` and `{table_candidate.right.table_name}` onto one authoritative row family.\n"
                "# Keep one writable axis table and derive any module-local indexes or views from it."
            ),
            metrics=MappingMetrics(
                mapping_site_count=2,
                field_count=max(len(table_candidate.shared_case_names), 1),
                mapping_name=table_candidate.left.table_name,
                field_names=table_candidate.shared_case_names,
                source_name=table_candidate.key_type_name,
                identity_field_names=("key",),
            ),
        )


class ParallelKeyedTableAndFamilyDetector(CrossModuleCandidateDetector):
    detector_id = "parallel_keyed_table_and_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Keyed table and keyed family should collapse into one auto-registered axis family",
        why=(
            "The docs require one authoritative owner per closed semantic axis. When a module keeps one keyed table of "
            "per-case records and a second keyed nominal family over the same cases, the axis is split across data and behavior. "
            "If the family already carries the runtime behavior boundary, the table should derive from that family instead of competing with it."
        ),
        capability_gap="single authoritative metaclass-registry axis family with derived table/view projections",
        relation_context="same enum/key axis is encoded by both a keyed table and a keyed nominal family",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.AUTHORITATIVE_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _parallel_keyed_table_and_family_candidates(modules)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        table_candidate = cast(ParallelKeyedTableAndFamilyCandidate, candidate)
        shape_clause = (
            ""
            if table_candidate.value_shape_name is None
            else f" of `{table_candidate.value_shape_name}` records"
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Axis `{table_candidate.key_type_name}` is split between keyed table `{table_candidate.table_name}`"
                f"{shape_clause} and keyed family `{table_candidate.family_name}` across cases "
                f"{', '.join(table_candidate.shared_case_names[:4])}."
            ),
            table_candidate.evidence,
            scaffold=(
                _axis_policy_registry_scaffold("build(self)")
                + "\n\n"
                "@dataclass(frozen=True)\n"
                "class DerivedAxisRow:\n"
                f"    key: {_AXIS_POLICY_KEY_TYPE_NAME}\n"
                f"    policy_type: type[{_AXIS_POLICY_ROOT_NAME}]\n"
                "    config: object\n\n"
                "def build_axis_rows() -> tuple[DerivedAxisRow, ...]:\n"
                "    return tuple(\n"
                "        DerivedAxisRow(key=key, policy_type=policy_type, config=...)\n"
                f"        for key, policy_type in {_AXIS_POLICY_ROOT_NAME}.__registry__.items()\n"
                "    )"
            ),
            codemod_patch=(
                f"# Collapse `{table_candidate.table_name}` and `{table_candidate.family_name}` onto one authoritative metaclass-registry family.\n"
                "# Keep the runtime boundary on the auto-registered family and derive any keyed rows/views from `AxisPolicy.__registry__`."
            ),
            metrics=_axis_dispatch_metrics(
                table_candidate.shared_case_names,
                table_candidate.key_type_name,
            ),
        )


class EnumKeyedTableClassAxisShadowDetector(CandidateFindingDetector):
    detector_id = "enum_keyed_table_class_axis_shadow"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Enum-keyed table should derive from auto-registered class-declared axis keys",
        why=(
            "The docs require a single writable owner per closed semantic axis. If a module already declares "
            "that axis through class-level enum assignments, adding a writable enum-keyed table over the same "
            "cases creates duplicate authority and a synchronization surface. The class-declared axis should be the "
            "primary owner and any enum-keyed lookup should be derived from the family registry."
        ),
        capability_gap="one authoritative metaclass-registry closed-axis owner with derived table/view projections",
        relation_context="module-level enum-keyed table overlaps a class family that already declares the same enum axis",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.PROJECTION_DICT,
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _enum_keyed_table_class_axis_shadow_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        axis_candidate = cast(EnumKeyedTableClassAxisShadowCandidate, candidate)
        class_names = ", ".join(axis_candidate.class_names[:4])
        shared_cases = ", ".join(axis_candidate.shared_case_names[:4])
        value_names = ", ".join(axis_candidate.value_type_names[:4])
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{axis_candidate.table_name}` maps `{axis_candidate.key_type_name}` cases {shared_cases} "
                f"to {value_names}, while classes {class_names} already declare the same axis via "
                f"`{axis_candidate.key_attr_name}`."
            ),
            axis_candidate.evidence,
            scaffold=(
                _axis_policy_registry_scaffold("route_type(self)")
                + "\n\n"
                "AXIS_BY_KEY = {\n"
                "    key: policy_type\n"
                f"    for key, policy_type in {_AXIS_POLICY_ROOT_NAME}.__registry__.items()\n"
                "}\n"
            ),
            codemod_patch=(
                f"# Remove `{axis_candidate.table_name}` as a second writable authority.\n"
                f"# Derive `{axis_candidate.key_type_name}` lookup views from the auto-registered family keyed by `{axis_candidate.key_attr_name}` instead of hardcoding a parallel table."
            ),
            metrics=MappingMetrics(
                mapping_site_count=len(axis_candidate.shared_case_names),
                field_count=1,
                mapping_name=axis_candidate.table_name,
                field_names=(axis_candidate.key_attr_name,),
                source_name=axis_candidate.key_type_name,
                identity_field_names=(axis_candidate.key_attr_name,),
            ),
        )


class TransportShellTemplateMethodDetector(CandidateFindingDetector):
    detector_id = "transport_shell_template_method"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Template-method family is a transport shell over a downstream authority",
        why=(
            "The docs say nominal families should have one authoritative owner. When an ABC template method only "
            "materializes an intermediate object from a class-level selector, delegates through one hook, and "
            "repackages through another hook, the extra family is usually a transport shell around an already "
            "authoritative boundary."
        ),
        capability_gap="single authoritative materialization/execution family instead of a parallel transport shell",
        relation_context="template family varies mostly by class-level selector and result adapter",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _transport_shell_template_candidates(module, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        shell_candidate = cast(TransportShellTemplateCandidate, candidate)
        selector_values = ", ".join(shell_candidate.selector_value_names)
        kwargs_clause = (
            f" plus `{shell_candidate.kwargs_helper_name}({shell_candidate.source_param_name})`"
            if shell_candidate.kwargs_helper_name is not None
            else ""
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{shell_candidate.class_name}.{shell_candidate.driver_method_name}` materializes selector values "
                f"{selector_values} from `{shell_candidate.selector_attr_name}` via `{shell_candidate.constructor_name}`"
                f"{kwargs_clause} across {len(shell_candidate.concrete_class_names)} concrete leaves, then only delegates "
                f"through `{shell_candidate.inner_hook_name}` and `{shell_candidate.outer_hook_name}`."
            ),
            (shell_candidate.evidence,),
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class MaterializationSpec:\n"
                "    selector: object\n"
                "    materializer: object\n"
                "    executor: object\n"
                "    packager: object\n"
                "# Dispatch once on the authoritative selector/spec family."
            ),
            codemod_patch=(
                f"# Collapse `{shell_candidate.class_name}` onto the downstream selector/spec family.\n"
                "# Keep one selection boundary and let that boundary own materialization, execution, and result packaging."
            ),
        )


class CrossModuleSpecAxisAuthorityDetector(CrossModuleCandidateDetector):
    detector_id = "cross_module_spec_axis_authority"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Cross-module spec axis should have one authority",
        why=(
            "The docs say one semantic family should have one authoritative owner. When two modules encode the same "
            "identity-axis -> executable-axis spec pairs, one table is a duplicate authority unless it is explicitly derived."
        ),
        capability_gap="one repository-wide authoritative spec-axis family",
        relation_context="same identity/executable spec axis is re-encoded across modules",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.CLASS_FAMILY,
        ),
    )

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[object]:
        return _cross_module_spec_axis_authority_candidates(modules, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        authority_candidate = cast(CrossModuleSpecAxisAuthorityCandidate, candidate)
        family_names = ", ".join(
            f"{Path(family.file_path).name}:{family.family_name}"
            for family in authority_candidate.families
        )
        pair_names = ", ".join(
            f"{identity}->{executable}"
            for identity, executable in authority_candidate.shared_axis_pairs
        )
        axis_fields = " -> ".join(authority_candidate.axis_field_names)
        evidence = tuple(
            family.evidence for family in authority_candidate.families[:6]
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Families {family_names} each encode the same `{axis_fields}` pairs {pair_names} across module boundaries."
            ),
            evidence,
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class AxisExecutionSpec:\n"
                "    identity: object\n"
                "    executable: object\n"
                "# Keep one exported authority and let downstream modules compose from it."
            ),
            codemod_patch=(
                "# Extract one repository-wide spec-axis family.\n"
                "# Make downstream wrappers, benchmarks, or adapters reference that authority instead of restating identity/executable pairs."
            ),
        )


class ParallelRegistryProjectionFamilyDetector(CandidateFindingDetector):
    detector_id = "parallel_registry_projection_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Parallel registry projection builders should collapse into one family spec",
        why=(
            "The docs say one semantic family should have one authoritative owner. When several functions differ only in "
            "which registry authority feeds which target constructor, the projection-axis mapping should become one declared "
            "spec or family authority instead of several hand-wired wrappers."
        ),
        capability_gap="single authoritative registry-projection family",
        relation_context="same registry-authority-to-target projection shape repeated across sibling functions",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
        observation_tags=(
            ObservationTag.BUILDER_CALL,
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _parallel_registry_projection_family_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        catalog_candidate = cast(ParallelRegistryProjectionFamilyCandidate, candidate)
        function_names = ", ".join(
            function.qualname for function in catalog_candidate.functions[:4]
        )
        extractor_bases = ", ".join(
            function.extractor_base_name for function in catalog_candidate.functions[:4]
        )
        catalog_types = ", ".join(
            function.catalog_type_name for function in catalog_candidate.functions[:4]
        )
        evidence = tuple(function.evidence for function in catalog_candidate.functions[:6])
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Functions {function_names} each build {catalog_types} through "
                f"`{catalog_candidate.collector_name}(structure, ExtractorBase.{catalog_candidate.registry_accessor_name}())` "
                f"over parallel extractor bases {extractor_bases}."
            ),
            evidence,
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class RegistryProjectionSpec:\n"
                "    registry_authority: type\n"
                "    target_type: type\n"
                "# One helper should own the registry-authority to target mapping."
            ),
            codemod_patch=(
                "# Extract one registry-projection family spec and one authoritative projection builder.\n"
                "# Make per-axis public helpers delegate to that authority instead of reconstructing collector(...registry_accessor())."
            ),
        )


class RepeatedKeyedFamilyDetector(CrossModuleCandidateDetector):
    detector_id = "repeated_keyed_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTO_REGISTER_META,
        title="Repeated keyed family scaffolding should collapse into one typed metaclass-registry base",
        why=(
            "The docs encourage aggressive metaprogramming when several nominal families repeat the same "
            "class-level registration and lookup shell. When many roots restate `registry_key_attr`, "
            "`_registry`, and `for_*` lookup methods, the family algorithm should live in one typed "
            "`metaclass-registry` base."
        ),
        capability_gap="single typed metaclass-registry substrate for keyed nominal registries",
        relation_context="same keyed family registration and lookup shell repeated across nominal family roots",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.ENUMERATION,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
        ),
    )

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[object]:
        return _repeated_keyed_family_candidates(modules, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        family_candidate = cast(RepeatedKeyedFamilyCandidate, candidate)
        class_names = ", ".join(
            root.class_name for root in family_candidate.roots[:8]
        )
        lookup_names = ", ".join(
            sorted({root.lookup_method_name for root in family_candidate.roots[:8]})
        )
        registry_keys = ", ".join(
            sorted({root.registry_key_attr_name for root in family_candidate.roots[:8]})
        )
        evidence = tuple(root.evidence for root in family_candidate.roots[:8])
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Registry roots {class_names} each repeat `{registry_keys}` + `_registry` + "
                f"`{lookup_names}` over `{family_candidate.family_base_name}`."
            ),
            evidence,
            scaffold=(
                "from metaclass_registry import AutoRegisterMeta\n\n"
                "KeyT = TypeVar(\"KeyT\")\n\n"
                "class KeyedNominalFamily(ABC, Generic[KeyT], metaclass=AutoRegisterMeta):\n"
                "    __registry_key__ = \"registry_key\"\n"
                "    __skip_if_no_key__ = True\n"
                "    registry_key: ClassVar[KeyT | None] = None\n"
                "    family_label: ClassVar[str] = \"family\"\n"
                "    @classmethod\n"
                "    def for_key(cls, key: KeyT):\n"
                "        try:\n"
                "            return cls.__registry__[key]\n"
                "        except KeyError as error:\n"
                "            raise ValueError(f\"Unknown {cls.family_label}: {key}\") from error"
            ),
            codemod_patch=(
                "# Extract one typed metaclass-registry base that owns registration lookup, duplicate handling, and error shaping.\n"
                "# Leave only declarative key attributes and irreducible hook methods on each family root, and read the registered classes from `cls.__registry__`."
            ),
        )


class ManualKeyedRecordTableDetector(CandidateFindingDetector):
    detector_id = "manual_keyed_record_table"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Manual keyed record tables should collapse into one authoritative spec table",
        why=(
            "When several frozen record classes repeat `_registry`, `register`, and `for_*` lookup around closed keys, "
            "the code is hand-maintaining multiple writable tables. The docs prefer one authoritative spec tuple or "
            "generic keyed-record table with derived indexes."
        ),
        capability_gap="single authoritative keyed-record table or derived index",
        relation_context="same manual record registration and keyed lookup shell repeated across data classes",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.CLOSED_FAMILY_DISPATCH,
            CapabilityTag.PROVENANCE,
        ),
        observation_tags=(
            ObservationTag.BUILDER_CALL,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.CLASS_FAMILY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _manual_keyed_record_table_group_candidates(module, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        group_candidate = cast(ManualKeyedRecordTableGroupCandidate, candidate)
        class_names = ", ".join(
            item.class_name for item in group_candidate.classes[:6]
        )
        key_fields = ", ".join(
            sorted({item.key_field_name for item in group_candidate.classes[:6]})
        )
        lookup_names = ", ".join(
            sorted({item.lookup_method_name for item in group_candidate.classes[:6]})
        )
        evidence = tuple(item.evidence for item in group_candidate.classes[:6])
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Record tables {class_names} each repeat `_registry`, `{group_candidate.classes[0].register_method_name}`, "
                f"and `{lookup_names}` around key fields {key_fields}."
            ),
            evidence,
            scaffold=(
                "KeyT = TypeVar(\"KeyT\")\n"
                "RecordT = TypeVar(\"RecordT\")\n\n"
                "@dataclass(frozen=True)\n"
                "class KeyedRecordTable(Generic[KeyT, RecordT]):\n"
                "    records: tuple[RecordT, ...]\n"
                "    key_of: Callable[[RecordT], KeyT]\n\n"
                "    def by_key(self) -> dict[KeyT, RecordT]:\n"
                "        return {self.key_of(record): record for record in self.records}"
            ),
            codemod_patch=(
                "# Replace per-class mutable `_registry` + `register` shells with one authoritative tuple of record specs.\n"
                "# Derive the keyed lookup dict once, or factor the pattern into a generic keyed-record table helper."
            ),
        )


class ManualStructuralRecordMechanicsDetector(CandidateFindingDetector):
    detector_id = "manual_structural_record_mechanics"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_SCHEMA,
        title="Repeated structural record mechanics should derive from field metadata",
        why=(
            "When several frozen dataclass records hand-write validation, tuple-style field projection, "
            "round-trip reconstruction, and fieldwise transform logic, those mechanics have become a second "
            "authority beside the field declarations. The docs prefer one metadata-driven record substrate "
            "that derives those mechanics from typed fields."
        ),
        capability_gap="single typed structural-record substrate with derived validation, projection, and transform mechanics",
        relation_context="same dataclass record lifecycle mechanics repeated across sibling structural record classes",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
            CapabilityTag.TYPE_LINEAGE,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.BUILDER_CALL,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _manual_structural_record_mechanics_group_candidates(module, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        group_candidate = cast(ManualStructuralRecordMechanicsGroupCandidate, candidate)
        class_names = ", ".join(
            item.class_name for item in group_candidate.classes[:6]
        )
        shared_methods = ", ".join(group_candidate.shared_method_names)
        transform_methods = ", ".join(group_candidate.transform_method_names[:6])
        base_names = ", ".join(group_candidate.base_names)
        evidence = tuple(item.evidence for item in group_candidate.classes[:6])
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Dataclass records {class_names} each hand-roll `{shared_methods}` plus fieldwise transforms "
                f"{transform_methods} on top of base family `{base_names}`."
            ),
            evidence,
            scaffold=(
                "@dataclass_transform(field_specifiers=(field, record_field))\n"
                "class StructuralRecordBase:\n"
                "    def validate(self): ...\n"
                "    def project_fields(self): ...\n"
                "    @classmethod\n"
                "    def from_projected(cls, projected, metadata): ...\n"
                "    def transformed(self, **changes): ...\n"
            ),
            codemod_patch=(
                "# Move validation constraints, projected-field partitions, and transform semantics into typed field metadata.\n"
                "# Derive projection, round-trip reconstruction, and fieldwise transforms from one structural-record base instead of re-encoding them per class."
            ),
        )


class RepeatedConcreteTypeCaseAnalysisDetector(CrossModuleCandidateDetector):
    detector_id = "repeated_concrete_type_case_analysis"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_INTERFACE_WITNESS,
        title="Repeated concrete-type recovery should become nominal family behavior",
        why=(
            "When several functions repeatedly recover the same semantic family through concrete `isinstance` "
            "checks on one carried attribute, the family boundary is still latent. The docs want one nominal "
            "ABC and concrete leaf behavior exposed through typed properties or hooks instead of repeated leaf decoding."
        ),
        capability_gap="single ABC-backed family for the carried subject, with repeated case recovery moved into nominal properties or hooks",
        relation_context="same attribute-carried family is re-decoded through repeated concrete runtime type checks across several functions",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.MRO_ORDERING,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.PARTIAL_VIEW,
        ),
    )

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[object]:
        return _repeated_concrete_type_case_analysis_candidates(modules, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        case_candidate = cast(RepeatedConcreteTypeCaseAnalysisCandidate, candidate)
        function_names = ", ".join(
            function.function_name for function in case_candidate.functions[:6]
        )
        class_names = ", ".join(case_candidate.concrete_class_names[:6])
        alias_summary = (
            f" Union alias(es): {', '.join(case_candidate.union_alias_names)}."
            if case_candidate.union_alias_names
            else ""
        )
        existing_base_summary = (
            f" Existing abstract base(s): {', '.join(case_candidate.abstract_base_names)}."
            if case_candidate.abstract_base_names
            else ""
        )
        suggested_family_name = _camel_case(case_candidate.subject_role)
        shared_suffix = _longest_common_suffix(case_candidate.concrete_class_names)
        if (
            shared_suffix
            and len(shared_suffix) >= 6
            and not suggested_family_name.endswith(shared_suffix)
        ):
            suggested_family_name = f"{suggested_family_name}{shared_suffix}"
        elif not suggested_family_name.endswith(("Family", "Witness", "Variant")):
            suggested_family_name = f"{suggested_family_name}Family"
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Functions {function_names} repeatedly recover `{case_candidate.subject_role}` across concrete classes {class_names}.{alias_summary}{existing_base_summary}"
            ),
            case_candidate.evidence,
            scaffold=(
                f"class {suggested_family_name}(ABC):\n"
                "    @property\n"
                "    @abstractmethod\n"
                "    def case_label(self) -> str: ...\n\n"
                "    def explain_case(self, context):\n"
                "        return None\n"
            ),
            codemod_patch=(
                f"# Type `{case_candidate.subject_role}` against one nominal ABC family instead of a concrete union surface.\n"
                "# Move repeated concrete `isinstance` recovery into abstract properties or case hooks on that family.\n"
                "# Keep only irreducible case-local residue in the concrete subclasses."
            ),
            metrics=DispatchCountMetrics(
                dispatch_site_count=len(case_candidate.functions),
                dispatch_axis=case_candidate.subject_role,
                literal_cases=case_candidate.concrete_class_names,
            ),
        )


class ImplicitSelfContractMixinDetector(CrossModuleCandidateDetector):
    detector_id = "implicit_self_contract_mixin"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Concrete mixins should not hide consumer contracts behind `self`-casts",
        why=(
            "The docs reserve mixins for orthogonal reusable concerns that participate in nominal MRO cleanly. "
            "When a concrete mixin erases `self` through `cast(..., self)` to reach consumer-owned fields, the "
            "mixin is carrying non-orthogonal family logic through a hidden contract instead of a declared base or policy."
        ),
        capability_gap="declared nominal base or policy row for the shared algorithm instead of a hidden mixin self-contract",
        relation_context="concrete mixin methods erase `self` through casts and depend on consumer-owned attributes across several subclasses",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.MRO_ORDERING,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.REPEATED_METHOD_ROLES,
            ObservationTag.PARTIAL_VIEW,
        ),
    )

    def _candidate_items(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> Sequence[object]:
        return _implicit_self_contract_mixin_candidates(modules, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        mixin_candidate = cast(ImplicitSelfContractMixinCandidate, candidate)
        methods = ", ".join(mixin_candidate.method_names)
        consumers = ", ".join(mixin_candidate.consumer_class_names[:6])
        accessed_attributes = ", ".join(mixin_candidate.accessed_attribute_names[:6])
        cast_types = ", ".join(mixin_candidate.cast_type_names[:6])
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{mixin_candidate.mixin_name}` uses `cast(..., self)` ({cast_types}) in `{methods}` to reach consumer-owned attributes ({accessed_attributes}) across subclasses {consumers}."
            ),
            mixin_candidate.evidence,
            scaffold=(
                "class FamilyBase(ABC):\n"
                "    def run_shared_step(self): ...\n\n"
                "class CasePolicy(ABC):\n"
                "    def run(self, request): ...\n"
            ),
            codemod_patch=(
                f"# `{mixin_candidate.mixin_name}` is not an orthogonal mixin; it hides a consumer contract behind `cast(..., self)`.\n"
                "# Move the shared behavior to a declared nominal base or a keyed policy/spec family, and leave only true orthogonal residue in mixins."
            ),
            metrics=HierarchyCandidateMetrics(
                duplicate_group_count=len(mixin_candidate.method_names),
                class_count=len(mixin_candidate.consumer_class_names) + 1,
            ),
        )


class RepeatedGuardValidatorFamilyDetector(CandidateFindingDetector):
    detector_id = "repeated_guard_validator_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Repeated guard validators should collapse into one case-policy authority",
        why=(
            "When several sibling boolean helpers walk the same subject through fail-fast guards and case-local final "
            "checks, the algorithm skeleton is split across helper names instead of being owned by one nominal case "
            "policy or declarative rule family."
        ),
        capability_gap="single authoritative case-policy or rule-table validator",
        relation_context="same subject and subordinate view validated through repeated fail-fast sibling helpers",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.AUTHORITATIVE_MAPPING,
        ),
        observation_tags=(
            ObservationTag.DATAFLOW_ROOT,
            ObservationTag.PARTIAL_VIEW,
            ObservationTag.CLASS_FAMILY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _repeated_guard_validator_family_candidates(module, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        family_candidate = cast(RepeatedGuardValidatorFamilyCandidate, candidate)
        function_names = ", ".join(
            function.function_name for function in family_candidate.functions[:6]
        )
        shared_attrs = ", ".join(family_candidate.shared_attr_names[:6])
        alias_summary = (
            f" through `{family_candidate.alias_source_attr}`"
            if family_candidate.alias_source_attr is not None
            else ""
        )
        shared_helpers = ", ".join(family_candidate.shared_helper_call_names[:3])
        helper_summary = (
            f" Shared helper calls: {shared_helpers}."
            if shared_helpers
            else ""
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Boolean validators {function_names} each guard `{family_candidate.subject_param_name}`{alias_summary} "
                f"with the same fail-fast attribute checks over {shared_attrs}.{helper_summary}"
            ),
            family_candidate.evidence,
            scaffold=(
                "class ValidationCasePolicy(ABC):\n"
                "    def validation_error(self, subject):\n"
                "        child = self._subject_child(subject)\n"
                "        if not self._shared_preconditions(subject, child):\n"
                "            return self._shared_failure_message()\n"
                "        return self._case_specific_error(subject, child)\n\n"
                "    @abstractmethod\n"
                "    def _case_specific_error(self, subject, child): ..."
            ),
            codemod_patch=(
                "# Collapse these sibling boolean helpers into one authoritative case-policy family or one declarative rule table.\n"
                "# Keep shared fail-fast guards in one concrete validator method, and leave only case-specific predicates or handle sets per case."
            ),
        )


class RepeatedValidateShapeGuardFamilyDetector(IssueDetector):
    detector_id = "repeated_validate_shape_guard_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Repeated validate() shape guards should collapse into one validated-record authority",
        why=(
            "Sibling nominal records repeat the same fail-fast shape and dimensional guards in `validate()` while "
            "differing only in field names or a small residue check. The docs treat that as duplicated contract "
            "authority that should move into one shared validated-record base, field-spec table, or mixin hook."
        ),
        capability_gap="single authoritative validated-record contract for repeated shape/ndim guards",
        relation_context="same nominal record family repeats fail-loud shape validation scaffolding",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.AUTHORITATIVE_MAPPING,
        ),
        observation_tags=(
            ObservationTag.CLASS_FAMILY,
            ObservationTag.METHOD_ROLE,
            ObservationTag.NORMALIZED_AST,
        ),
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        return [
            self._finding_for_candidate(candidate)
            for candidate in _repeated_validate_shape_guard_candidates_for_modules(
                modules, config
            )
        ]

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        family_candidate = cast(RepeatedValidateShapeGuardFamilyCandidate, candidate)
        method_symbols = tuple(method.symbol for method in family_candidate.methods)
        method_summary = ", ".join(method_symbols[:6])
        shared_guard_count = len(family_candidate.shared_shape_guard_signatures)
        shared_guard_preview = ", ".join(
            family_candidate.shared_shape_guard_signatures[:3]
        )
        preview_suffix = (
            f" Shared normalized guards include {shared_guard_preview}."
            if shared_guard_preview
            else ""
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Validate methods {method_summary} repeat {shared_guard_count} shared shape/ndim guard forms."
            ),
            family_candidate.evidence,
            scaffold=(
                "class ShapeValidatedRecord(ABC):\n"
                "    def validate(self):\n"
                "        for predicate, message in self._shape_guard_rules():\n"
                "            if predicate(self):\n"
                "                raise ValueError(message)\n"
                "        self._validate_residue()\n\n"
                "    @classmethod\n"
                "    @abstractmethod\n"
                "    def _shape_guard_rules(cls): ...\n\n"
                "    def _validate_residue(self):\n"
                "        return None"
                f"{preview_suffix}"
            ),
            codemod_patch=(
                "# Collapse repeated `validate()` shape guards into one authoritative validated-record base or field-spec table.\n"
                "# Keep only the truly variable residue checks, messages, or field roster on each concrete record."
            ),
        )


class RepeatedResultAssemblyPipelineDetector(CandidateFindingDetector):
    detector_id = "repeated_result_assembly_pipeline"
    finding_spec = FindingSpec(
        pattern_id=PatternId.ABC_TEMPLATE_METHOD,
        title="Repeated result-assembly pipeline should collapse into one authoritative assembler",
        why=(
            "Several owners repeat the same downstream result-assembly stages and differ only in the "
            "upstream source or projection that feeds the pipeline. The docs treat that as shared "
            "algorithm authority that should move into one template method or authoritative helper with "
            "one orthogonal source hook."
        ),
        capability_gap="single authoritative result-assembly pipeline with one source hook",
        relation_context="same staged assembly tail is repeated across sibling functions or methods",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.SHARED_ALGORITHM_AUTHORITY,
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _repeated_result_assembly_pipeline_candidates(module, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        pipeline_candidate = cast(RepeatedResultAssemblyPipelineCandidate, candidate)
        function_names = ", ".join(
            function.qualname for function in pipeline_candidate.functions[:4]
        )
        stage_names = ", ".join(
            stage.callee_name for stage in pipeline_candidate.shared_tail
        )
        evidence = tuple(
            function.evidence for function in pipeline_candidate.functions[:6]
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Functions {function_names} share the same result-assembly tail "
                f"{stage_names} and differ only in their leading source stages."
            ),
            evidence,
            scaffold=(
                "class ResultAssembler(ABC):\n"
                "    @abstractmethod\n"
                "    def supply_inputs(self, request): ...\n\n"
                "    def assemble(self, request):\n"
                "        supplied = self.supply_inputs(request)\n"
                "        # run the shared downstream assembly stages here\n"
                "        return result"
            ),
            codemod_patch=(
                "# Extract the shared assignment/return tail into one authoritative helper.\n"
                "# Leave only the source-supplier stage variant-specific."
            ),
            metrics=RepeatedMethodMetrics.from_duplicate_family(
                duplicate_site_count=len(pipeline_candidate.functions),
                statement_count=len(pipeline_candidate.shared_tail),
                class_count=len(
                    {
                        function.qualname.split(".", 1)[0]
                        for function in pipeline_candidate.functions
                        if "." in function.qualname
                    }
                    or {pipeline_candidate.functions[0].qualname}
                ),
                method_symbols=tuple(
                    function.qualname for function in pipeline_candidate.functions
                ),
                shared_statement_texts=tuple(
                    stage.callee_name for stage in pipeline_candidate.shared_tail
                ),
            ),
        )


class NestedBuilderShellDetector(CandidateFindingDetector):
    detector_id = "nested_builder_shell"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTHORITATIVE_CONTEXT,
        title="Nested builder shell should collapse into one authoritative request boundary",
        why=(
            "A builder forwards a substantial semantic parameter family unchanged into a subordinate "
            "nominal builder and only adds a small residue locally. The docs treat that as split request "
            "authority: one layer should own the forwarded family instead of rebuilding it inside another shell."
        ),
        capability_gap="single authoritative request/context builder boundary",
        relation_context="one builder nests a forwarded subordinate request builder inside a second nominal shell",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.PROVENANCE,
            CapabilityTag.UNIT_RATE_COHERENCE,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return _nested_builder_shell_candidates(module, config)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        shell_candidate = cast(NestedBuilderShellCandidate, candidate)
        forwarded = ", ".join(shell_candidate.forwarded_parameter_names)
        residue_fields = ", ".join(shell_candidate.residue_field_names)
        residue_sources = ", ".join(shell_candidate.residue_source_names)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{shell_candidate.qualname}` forwards `{forwarded}` into "
                f"`{shell_candidate.nested_callee_name}` under `{shell_candidate.nested_field_name}` "
                f"while separately deriving `{residue_fields}` from `{residue_sources}`."
            ),
            (shell_candidate.evidence,),
            scaffold=(
                "@dataclass(frozen=True)\n"
                "class OuterRequest:\n"
                "    child_request: ChildRequest\n\n"
                "    @classmethod\n"
                "    def from_source(cls, source, *, child_request: ChildRequest):\n"
                "        return cls(child_request=child_request, ...)\n"
            ),
            codemod_patch=(
                f"# Stop rebuilding `{shell_candidate.nested_callee_name}` inside `{shell_candidate.qualname}`.\n"
                "# Accept the subordinate request/context directly, or move both layers into one authoritative builder."
            ),
            metrics=ParameterThreadMetrics(
                function_count=1,
                shared_parameter_count=len(shell_candidate.forwarded_parameter_names),
                shared_parameter_names=shell_candidate.forwarded_parameter_names,
            ),
        )


class ManualFiberTagDetector(CandidateFindingDetector):
    detector_id = "manual_fiber_tag"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_BOUNDARY,
        title="Manual fiber tag should become nominal family",
        why=(
            "A string-valued instance tag is manually selecting behavior while the same instance still carries fields from several incompatible fibers. "
            "That leaves the family above the zero-incoherence threshold and admits disagreement states the host type system could rule out."
        ),
        capability_gap="host-native nominal fiber decomposition with one subclass per behavior fiber",
        relation_context="manual instance tag drives behavior while irrelevant coordinates remain constructible on every fiber",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _manual_fiber_tag_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        fiber_candidate = cast(ManualFiberTagCandidate, candidate)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{fiber_candidate.class_name}` branches on manual fiber tag `self.{fiber_candidate.tag_name}` across {fiber_candidate.case_names} while still carrying cross-fiber fields {fiber_candidate.assigned_field_names}."
            ),
            (
                SourceLocation(
                    fiber_candidate.file_path,
                    fiber_candidate.init_line,
                    f"{fiber_candidate.class_name}.__init__",
                ),
                SourceLocation(
                    fiber_candidate.file_path,
                    fiber_candidate.method_line,
                    f"{fiber_candidate.class_name}.{fiber_candidate.method_name}",
                ),
            ),
            scaffold=_manual_fiber_tag_scaffold(fiber_candidate),
            codemod_patch=_manual_fiber_tag_patch(fiber_candidate),
            metrics=DispatchCountMetrics(
                dispatch_site_count=len(fiber_candidate.case_names),
                dispatch_axis=f"self.{fiber_candidate.tag_name}",
                literal_cases=fiber_candidate.case_names,
            ),
        )


class DescriptorDerivedViewDetector(CandidateFindingDetector):
    detector_id = "descriptor_derived_view"
    finding_spec = FindingSpec(
        pattern_id=PatternId.DESCRIPTOR_DERIVED_VIEW,
        title="Derived views stored independently of their source",
        why=(
            "Several stored fields are derived from one authoritative source field, but mutators resynchronize them manually and incompletely. "
            "That raises the degree of freedom above one and makes view disagreement reachable."
        ),
        capability_gap="descriptor- or property-mediated derived views rooted in one authoritative source",
        relation_context="stored derived views must be manually kept coherent with a single source field",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.AUTHORITATIVE_MAPPING,
            CapabilityTag.UNIT_RATE_COHERENCE,
            CapabilityTag.PROVENANCE,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _descriptor_derived_view_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        view_candidate = cast(DescriptorDerivedViewCandidate, candidate)
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{view_candidate.class_name}` stores derived views {view_candidate.derived_field_names} from `{view_candidate.source_attr}`, but `{view_candidate.mutator_name}` only updates {view_candidate.updated_field_names}."
            ),
            (
                SourceLocation(
                    view_candidate.file_path,
                    view_candidate.init_line,
                    f"{view_candidate.class_name}.__init__",
                ),
                SourceLocation(
                    view_candidate.file_path,
                    view_candidate.mutator_line,
                    f"{view_candidate.class_name}.{view_candidate.mutator_name}",
                ),
            ),
            scaffold=_descriptor_derived_view_scaffold(view_candidate),
            codemod_patch=_descriptor_derived_view_patch(view_candidate),
        )


class DeferredClassRegistrationDetector(CandidateFindingDetector):
    detector_id = "deferred_class_registration"
    finding_spec = FindingSpec(
        pattern_id=PatternId.AUTO_REGISTER_META,
        title="Class registration is decoupled from class existence",
        why=(
            "Manual decorator- or helper-based registration leaves a reachable state where a class exists but the registry has not been updated. "
            "The host already provides zero-delay registration via `metaclass-registry` or another class-time hook."
        ),
        capability_gap="zero-delay metaclass-registry class registration with collision checks and runtime provenance",
        relation_context="class registration is performed as a separate auxiliary step rather than at class creation time",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.CLASS_LEVEL_REGISTRATION,
            CapabilityTag.PROVENANCE,
            CapabilityTag.NOMINAL_IDENTITY,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _manual_registry_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        registry_candidate = cast(ManualRegistryCandidate, candidate)
        evidence = [
            SourceLocation(
                registry_candidate.file_path,
                registry_candidate.line,
                registry_candidate.decorator_name,
            ),
        ]
        evidence.extend(
            SourceLocation(
                registry_candidate.file_path,
                registry_candidate.line,
                class_name,
            )
            for class_name in registry_candidate.class_names[:5]
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Registry `{registry_candidate.registry_name}` is updated through manual decorator `{registry_candidate.decorator_name}` for classes {registry_candidate.class_names}, leaving registration structurally decoupled from class creation."
            ),
            tuple(evidence),
            scaffold=_manual_registry_scaffold(registry_candidate),
            codemod_patch=_manual_registry_patch(registry_candidate),
            metrics=RegistrationMetrics(
                registration_site_count=len(registry_candidate.class_names),
                registry_name=registry_candidate.registry_name,
            ),
        )


class StructuralConfusabilityDetector(CandidateFindingDetector):
    detector_id = "structural_confusability"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_INTERFACE_WITNESS,
        title="Consumer observes a confusable duck-typed family",
        why=(
            "A consumer only observes a partial structural view, and several unrelated classes are confusable under that view. "
            "Without a nominal witness, the distortion floor stays above zero and the family boundary remains implicit."
        ),
        capability_gap="ABC-backed nominal witness for a structurally confusable implementation family",
        relation_context="consumer depends on a partial structural view shared by several unrelated classes",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.FAIL_LOUD_CONTRACTS,
            CapabilityTag.PROVENANCE,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _structural_confusability_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        confusability_candidate = cast(StructuralConfusabilityCandidate, candidate)
        evidence = (
            SourceLocation(
                confusability_candidate.file_path,
                confusability_candidate.line,
                confusability_candidate.function_name,
            ),
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"`{confusability_candidate.function_name}` observes `{confusability_candidate.parameter_name}` only through methods {confusability_candidate.observed_method_names}, but classes {confusability_candidate.class_names} are confusable under that view."
            ),
            evidence,
            scaffold=_structural_confusability_scaffold(confusability_candidate),
            codemod_patch=_structural_confusability_patch(confusability_candidate),
        )


class SemanticWitnessFamilyDetector(CandidateFindingDetector):
    detector_id = "semantic_witness_family"
    finding_spec = FindingSpec(
        pattern_id=PatternId.NOMINAL_WITNESS_CARRIER,
        title="Semantic carrier family should share one nominal base",
        why=(
            "Several frozen dataclass carriers repeat the same location and naming roles under different field names. "
            "That leaves one semantic family structurally expanded instead of giving it one nominal carrier root."
        ),
        capability_gap="one authoritative nominal base for a semantic metadata carrier family",
        relation_context="same carrier family repeats a renamed semantic-role spine across sibling frozen dataclasses",
        confidence=HIGH_CONFIDENCE,
        certification=STRONG_HEURISTIC,
        capability_tags=(
            CapabilityTag.NOMINAL_IDENTITY,
            CapabilityTag.PROVENANCE,
            CapabilityTag.AUTHORITATIVE_MAPPING,
        ),
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        del config
        return _witness_carrier_family_candidates(module)

    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        witness_candidate = cast(WitnessCarrierFamilyCandidate, candidate)
        evidence = tuple(
            SourceLocation(witness_candidate.file_path, line, class_name)
            for class_name, line in zip(
                witness_candidate.class_names,
                witness_candidate.line_numbers,
                strict=True,
            )
        )
        return self.finding_spec.build(
            self.detector_id,
            (
                f"Frozen carrier classes {', '.join(witness_candidate.class_names)} repeat semantic roles {witness_candidate.shared_role_names} under renamed fields and should inherit one nominal base carrier."
            ),
            evidence,
            scaffold=_witness_carrier_family_scaffold(witness_candidate),
            codemod_patch=_witness_carrier_family_patch(witness_candidate),
            metrics=WitnessCarrierMetrics(
                class_count=len(witness_candidate.class_names),
                shared_role_count=len(witness_candidate.shared_role_names),
                class_names=witness_candidate.class_names,
                shared_role_names=witness_candidate.shared_role_names,
            ),
        )
