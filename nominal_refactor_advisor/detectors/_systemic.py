"""Systemic detector implementations.

This module holds the earlier detector families that focus on orchestration,
axis authority, registration, and other repo-wide architectural smells.
"""

from __future__ import annotations

from ..record_algebra import (
    materialize_product_record,
    materialize_product_records,
    product_record_spec,
)
from ..semantic_algebra import ObjectFamilyShape
from ..semantic_description_length import (
    ClassFamilyCompressionProfile,
    CompressionCertificate,
)

from ._base import *
from ._helpers import *
from ._helpers import _facade_only_nominal_authority_candidates

def _closed_axis_conversion_matrix_compression_certificate(
    candidate: ClosedAxisConversionMatrixCandidate,
) -> CompressionCertificate:
    return CompressionCertificate.from_object_family(
        manual_object_count=max(
            candidate.line_count + len(candidate.function_names),
            len(candidate.function_names) * 2,
        ),
        replacement_shape=ObjectFamilyShape(
            shared_objects=("conversion_dispatcher", "conversion_table"),
            per_axis_objects=("conversion_axis_case",),
        ),
        semantic_axes=(
            *(f"source:{item}" for item in candidate.source_axis_values),
            *(f"target:{item}" for item in candidate.target_axis_values),
        ),
    )


def _option_record_quotient_compression_certificate(
    candidate: OptionRecordQuotientCandidate,
) -> CompressionCertificate:
    return CompressionCertificate.from_object_family(
        manual_object_count=max(
            candidate.line_count,
            len(candidate.class_names) * max(len(candidate.field_names), 1),
        ),
        replacement_shape=ObjectFamilyShape(
            shared_objects=("option_schema_catalog",),
            per_axis_objects=("option_case",),
        ),
        semantic_axes=(*(f"record:{item}" for item in candidate.class_names),),
        residual_object_count=len(candidate.field_names)
        + len(candidate.default_names)
        + len(candidate.common_base_names),
    )


def _imported_name_aliases(
    module: ast.Module,
    *,
    module_names: frozenset[str],
    imported_name: str,
) -> frozenset[str]:
    aliases: set[str] = {imported_name}
    for node in module.body:
        if not isinstance(node, ast.ImportFrom) or node.module not in module_names:
            continue
        for alias in node.names:
            if alias.name == imported_name:
                aliases.add(alias.asname or alias.name)
    return frozenset(aliases)


def _module_aliases(module: ast.Module, module_names: frozenset[str]) -> frozenset[str]:
    aliases: set[str] = set()
    for node in module.body:
        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            if alias.name in module_names:
                aliases.add(alias.asname or alias.name.split(".", maxsplit=1)[0])
    return frozenset(aliases)


def _is_imported_name(expr: ast.AST, aliases: frozenset[str]) -> bool:
    return isinstance(expr, ast.Name) and expr.id in aliases


def _is_qualified_name(
    expr: ast.AST, *, module_aliases: frozenset[str], attr_name: str
) -> bool:
    return (
        isinstance(expr, ast.Attribute)
        and expr.attr == attr_name
        and isinstance(expr.value, ast.Name)
        and (expr.value.id in module_aliases)
    )


_SINGLE_TEMPLATE_CALL_METRICS = OrchestrationMetrics(
    function_line_count=0,
    branch_site_count=0,
    call_site_count=1,
    parameter_count=1,
    callee_family_count=1,
)


class TypingProtocolContractDetector(IssueDetector):
    detector_priority = -20
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Structural Protocol contract should be a nominal ABC",
        "`typing.Protocol` is structural: it lets values claim membership by shape rather than by a declared nominal contract. The advisor's nominal architecture rules should route those interfaces through ABCs, explicit subclassing, or ABC virtual registration instead.",
        "nominal runtime contract instead of structural shape membership",
        "class declares interface identity through structural typing",
        _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_VIRTUAL_MEMBERSHIP_CAPABILITY_TAGS,
        _CLASS_FAMILY_RUNTIME_MEMBERSHIP_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        typing_modules = frozenset({"typing", "typing_extensions"})
        for module in modules:
            protocol_aliases = _imported_name_aliases(
                module.module, module_names=typing_modules, imported_name="Protocol"
            )
            runtime_checkable_aliases = _imported_name_aliases(
                module.module,
                module_names=typing_modules,
                imported_name="runtime_checkable",
            )
            typing_aliases = _module_aliases(module.module, typing_modules)
            evidence: list[SourceLocation] = []
            protocol_class_names: list[str] = []
            for node in _walk_nodes(module.module):
                if not isinstance(node, ast.ClassDef):
                    continue
                inherits_protocol = any(
                    (
                        _is_imported_name(base, protocol_aliases)
                        or _is_qualified_name(
                            base, module_aliases=typing_aliases, attr_name="Protocol"
                        )
                        for base in node.bases
                    )
                )
                runtime_checkable = any(
                    (
                        _is_imported_name(decorator, runtime_checkable_aliases)
                        or _is_qualified_name(
                            decorator,
                            module_aliases=typing_aliases,
                            attr_name="runtime_checkable",
                        )
                        for decorator in node.decorator_list
                    )
                )
                if inherits_protocol:
                    protocol_class_names.append(node.name)
                    evidence.append(
                        SourceLocation(str(module.path), node.lineno, node.name)
                    )
                elif runtime_checkable:
                    evidence.append(
                        SourceLocation(
                            str(module.path),
                            node.lineno,
                            f"{node.name}.runtime_checkable",
                        )
                    )
            if not evidence:
                continue
            class_summary = ", ".join(protocol_class_names) or "runtime-checkable class"
            findings.append(
                self.build_finding(
                    (
                        f"{module.path} declares structural typing interfaces "
                        f"({class_summary}); replace them with ABC-backed nominal contracts."
                    ),
                    tuple(evidence[:8]),
                    scaffold=(
                        "from abc import ABC, abstractmethod\n\nclass ContractName(ABC):\n    @abstractmethod\n    def required_method(self, request): ...\n\n# Use direct subclassing for owned implementations, or `ContractName.register(ExternalType)` for explicit virtual membership."
                    ),
                )
            )
        return findings


class RepeatedPrivateMethodDetector(FiberCollectedShapeIssueDetector):
    detector_id = "repeated_private_methods"
    observation_kind = ObservationKind.METHOD_SHAPE
    finding_spec = high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Repeated non-orthogonal method skeleton across classes",
        "Shared orchestration logic is duplicated across a behavior family. The docs say this shared non-orthogonal logic should move into an ABC with a concrete template method, leaving only orthogonal hooks in subclasses.",
        "single authoritative algorithm for a nominal behavior family",
        "same method role across sibling classes",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS,
        _NORMALIZED_AST_CLASS_FAMILY_METHOD_ROLE_OBSERVATION_TAGS,
    )

    def _module_shapes(self, module: ParsedModule) -> tuple[object, ...]:
        return tuple(
            CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
                module, MethodShapeFamily, MethodShape
            )
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
        methods = sorted_tuple(
            (_as_method_shape(shape) for shape in shapes),
            key=lambda item: (item.file_path, item.lineno),
        )
        class_names = {method.class_name for method in methods}
        if len(methods) < 2 or len(class_names) < 2:
            return None
        method_names = {method.method_name for method in methods}
        if not methods[0].is_private and len(method_names) > 1:
            return None
        evidence = tuple(
            (
                SourceLocation(method.file_path, method.lineno, method.symbol)
                for method in methods[:6]
            )
        )
        relation = (
            "same private helper role across sibling classes"
            if methods[0].is_private
            else "same method role across sibling classes"
        )
        compression_profile = ClassFamilyCompressionProfile.from_repeated_method_family(
            class_count=len(class_names),
            shared_statement_count=methods[0].statement_count,
        )
        return self.build_finding(
            f"{len(methods)} methods across {len(class_names)} classes share the same normalized AST shape.",
            evidence,
            relation_context=relation,
            scaffold=_abc_scaffold_for_methods(methods),
            codemod_patch=_abc_patch_for_methods(methods),
            compression_certificate=compression_profile.compression_certificate,
            metrics=RepeatedMethodMetrics.from_duplicate_family(
                duplicate_site_count=len(methods),
                statement_count=methods[0].statement_count,
                class_count=len(class_names),
                method_symbols=tuple((method.symbol for method in methods)),
                shared_statement_texts=methods[0].statement_texts,
            ),
        )


class InheritanceHierarchyCandidateDetector(IssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Classes cluster into an ABC hierarchy candidate",
        "The same set of classes repeats multiple non-orthogonal method skeletons. The docs say this is a strong signal that the family should be factored into an ABC with one concrete template method layer; orthogonal reusable concerns can then live in mixins so MRO preserves declared precedence.",
        "single authoritative inheritance hierarchy for a duplicated behavior family",
        "same class set repeats several method roles across the same family boundary",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS,
        _REPEATED_METHOD_ROLES_CLASS_FAMILY_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        repeated_methods = tuple(
            (
                method
                for module in modules
                for method in CANDIDATE_COLLECTION_AUTHORITY.typed_family_items(
                    module, MethodShapeFamily, MethodShape
                )
                if method.class_name
                and method.statement_count >= config.min_duplicate_statements
            )
        )
        graph = ObservationGraph(
            tuple((method.structural_observation for method in repeated_methods))
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
                    (
                        _as_method_shape(item)
                        for item in SUPPORT_PROJECTION_AUTHORITY.materialize_observations(
                            fiber.observations, lookup
                        )
                    )
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
                self.build_finding(
                    f"Classes {', '.join(sorted(class_names))} share {len(groups)} repeated method-shape groups and repeated method roles that likely want one ABC family.",
                    tuple(evidence[:8]),
                    scaffold=_abc_family_scaffold(class_names, groups),
                    codemod_patch=_abc_family_patch(class_names, groups),
                    metrics=HierarchyCandidateMetrics(
                        duplicate_group_count=len(groups), class_count=len(class_names)
                    ),
                )
            )
        return findings


class OrchestrationHubDetector(CandidateFindingDetector[FunctionProfile]):
    finding_spec = high_confidence_spec(
        PatternId.STAGED_ORCHESTRATION,
        "Oversized orchestration hub",
        "One function is owning too many control branches, helper calls, and phase transitions at once. The architecture wants explicit staged boundaries so the orchestration surface remains nominal and legible.",
        "explicit staged orchestration boundaries with named phase contracts",
        "one owner centralizes many operational phases and helper families",
        _SHARED_ALGORITHM_AUTHORITY_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return tuple(
            (
                profile
                for profile in _function_profiles(module)
                if profile.line_count >= config.min_orchestration_function_lines
                and profile.branch_count >= config.min_orchestration_branches
                and (profile.call_count >= config.min_orchestration_calls)
            )
        )

    finding_renderer = CandidateFindingRenderer[FunctionProfile](
        summary=lambda profile: f"`{profile.qualname}` concentrates {profile.line_count} lines, {profile.branch_count} branches, and {profile.call_count} calls across {profile.callee_family_count} callee families in one owner.",
        evidence=lambda profile: (profile.evidence,),
        scaffold=lambda profile: _orchestration_stage_scaffold(profile),
        codemod_patch=lambda profile: _orchestration_stage_patch(profile),
        metrics=lambda profile: OrchestrationMetrics(
            function_line_count=profile.line_count,
            branch_site_count=profile.branch_count,
            call_site_count=profile.call_count,
            parameter_count=len(profile.parameter_names),
            callee_family_count=profile.callee_family_count,
        ),
    )


class BranchClusterUnderAbstractionDetector(
    CandidateFindingDetector[FunctionProfile]
):
    finding_spec = high_confidence_spec(
        PatternId.STAGED_ORCHESTRATION,
        "Branch cluster should be split into nominal stages",
        "A function can be under-abstracted even when it is not a full orchestration hub: many local branches mean multiple semantic cases or phases are being owned by one procedural surface. The normal form is named stages, nominal authorities, or strategy variants that each own one case family.",
        "nominal stage or authority boundaries instead of one branch-heavy owner",
        "one function concentrates many branch sites even without satisfying the larger orchestration-hub shape",
        _SHARED_ALGORITHM_AUTHORITY_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
    )

    def _candidate_items(
        self, module: ParsedModule, config: DetectorConfig
    ) -> Sequence[object]:
        return tuple(
            (
                profile
                for profile in _function_profiles(module)
                if profile.line_count >= config.min_branch_cluster_function_lines
                and profile.branch_count >= config.min_branch_cluster_branches
            )
        )

    finding_renderer = CandidateFindingRenderer[FunctionProfile](
        summary=lambda profile: f"`{profile.qualname}` has {profile.branch_count} branch sites over {profile.line_count} lines; split the case/phase surface behind nominal authorities before adding more conditions.",
        evidence=lambda profile: (profile.evidence,),
        scaffold=lambda profile: (
            f"class {profile.qualname.split('.')[-1].title().replace('_', '')}Stage(ABC):\n"
            "    @abstractmethod\n"
            "    def run(self, context): ...\n\n"
            "# Move each branch family into a named stage/authority; callers compose stages instead of branching inline."
        ),
        codemod_patch=lambda profile: (
            f"# Extract branch families from `{profile.qualname}` into named stage or authority classes.\n"
            "# Keep the original function as a thin coordinator that delegates through typed contracts."
        ),
        metrics=lambda profile: OrchestrationMetrics(
            function_line_count=profile.line_count,
            branch_site_count=profile.branch_count,
            call_site_count=profile.call_count,
            parameter_count=len(profile.parameter_names),
            callee_family_count=profile.callee_family_count,
        ),
    )


@dataclass(frozen=True)
class ClassRoleQuotientCandidate(ClassLineWitnessCandidate):
    method_count: int
    private_method_count: int
    public_method_count: int
    role_method_counts: tuple[tuple[str, int], ...]
    role_representatives: tuple[tuple[str, int, str], ...]
    self_state_attribute_count: int
    self_call_count: int
    cross_role_self_call_count: int

    @property
    def role_names(self) -> tuple[str, ...]:
        return tuple((role for role, _ in self.role_method_counts))

    @property
    def evidence_locations(self) -> tuple[SourceLocation, ...]:
        return tuple(
            (
                SourceLocation(self.file_path, line, f"{self.class_name}.{method_name}")
                for role, line, method_name in self.role_representatives
            )
        )


def _method_role_token(method_name: str) -> str:
    stripped = method_name.strip("_")
    if not stripped:
        return ""
    return stripped.split("_", maxsplit=1)[0]


def _class_role_quotient_candidates(
    module: ParsedModule,
) -> tuple[ClassRoleQuotientCandidate, ...]:
    candidates: list[ClassRoleQuotientCandidate] = []
    for class_node in (
        node for node in _walk_nodes(module.module) if isinstance(node, ast.ClassDef)
    ):
        methods = tuple(
            (
                statement
                for statement in class_node.body
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                and (
                    not (
                        statement.name.startswith("__")
                        and statement.name.endswith("__")
                    )
                )
            )
        )
        if not methods:
            continue
        role_groups: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = (
            defaultdict(list)
        )
        method_role: dict[str, str] = {}
        for method in methods:
            role = _method_role_token(method.name)
            if not role:
                continue
            role_groups[role].append(method)
            method_role[method.name] = role
        nontrivial_roles = {
            role: role_methods
            for role, role_methods in role_groups.items()
            if len(role_methods) >= 2
        }
        nontrivial_method_count = sum(
            (len(role_methods) for role_methods in nontrivial_roles.values())
        )
        private_method_count = sum(
            (1 for method in methods if method.name.startswith("_"))
        )
        public_method_count = len(methods) - private_method_count
        if len(nontrivial_roles) < 3:
            continue
        if private_method_count <= public_method_count:
            continue
        if nontrivial_method_count * 2 < len(methods):
            continue

        self_state_attributes: set[str] = set()
        self_call_count = 0
        cross_role_self_call_count = 0
        method_names = {method.name for method in methods}
        for method in methods:
            caller_role = method_role.get(method.name, "")
            for child in _walk_nodes(method):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id == "self"
                    and child.func.attr in method_names
                ):
                    self_call_count += 1
                    callee_role = method_role.get(child.func.attr, "")
                    if callee_role and caller_role and (callee_role != caller_role):
                        cross_role_self_call_count += 1
                elif (
                    isinstance(child, ast.Attribute)
                    and isinstance(child.value, ast.Name)
                    and child.value.id == "self"
                    and child.attr not in method_names
                ):
                    self_state_attributes.add(child.attr)

        role_method_counts = sorted_tuple(
            (
                (role, len(role_methods))
                for role, role_methods in nontrivial_roles.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        representatives: list[tuple[str, int, str]] = []
        for role, _ in role_method_counts:
            role_methods = sorted(
                nontrivial_roles[role], key=lambda method: method.lineno
            )
            representative = role_methods[0]
            representatives.append((role, representative.lineno, representative.name))

        candidates.append(
            ClassRoleQuotientCandidate(
                file_path=str(module.path),
                line=class_node.lineno,
                class_name=class_node.name,
                method_count=len(methods),
                private_method_count=private_method_count,
                public_method_count=public_method_count,
                role_method_counts=role_method_counts,
                role_representatives=tuple(representatives[:8]),
                self_state_attribute_count=len(self_state_attributes),
                self_call_count=self_call_count,
                cross_role_self_call_count=cross_role_self_call_count,
            )
        )
    return sorted_tuple(
        candidates,
        key=lambda candidate: (
            candidate.file_path,
            candidate.line,
            candidate.class_name,
        ),
    )


class ClassRoleQuotientDetector(
    ModuleCollectorCandidateDetector[ClassRoleQuotientCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.STAGED_ORCHESTRATION,
        "Class method-role quotient should become composed subsystems",
        "The class method namespace factors into several nontrivial role-equivalence classes, while private methods dominate the public facade. That quotient is a formal signal that one nominal object is carrying a product of subsystem algebras instead of composing role-owned services.",
        "composed subsystem authorities derived from the class method-role quotient",
        "one class contains several nontrivial method-role equivalence classes behind a smaller public facade",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS,
        _METHOD_ROLE_DATAFLOW_ROOT_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, role_candidate: ClassRoleQuotientCandidate
    ) -> RefactorFinding:
        role_summary = ", ".join(
            (f"{role}:{count}" for role, count in role_candidate.role_method_counts[:8])
        )
        return self.build_finding(
            (
                f"Class `{role_candidate.class_name}` has {role_candidate.method_count} methods "
                f"whose nontrivial method-role quotient is {role_summary}; "
                f"{role_candidate.private_method_count} private methods sit behind "
                f"{role_candidate.public_method_count} public methods."
            ),
            role_candidate.evidence_locations,
            scaffold=(
                "@dataclass(frozen=True)\nclass BuilderContext:\n    ...\n\nclass RoleSubsystem:\n    def __init__(self, context: BuilderContext): ...\n\nclass Facade:\n    def __init__(self, context: BuilderContext):\n        self.role = RoleSubsystem(context)\n\n# Factor the role quotient into composed subsystem authorities. Leave the original class as a public facade and sequencing boundary."
            ),
            codemod_patch=(
                "# Partition the class by the method-role quotient. Move each cohesive role class into a subsystem object or mixin with a shared context record.\n# The facade should expose public operations and delegate to the composed role authorities."
            ),
            metrics=OrchestrationMetrics(
                function_line_count=0,
                branch_site_count=role_candidate.private_method_count,
                call_site_count=role_candidate.self_call_count,
                parameter_count=role_candidate.self_state_attribute_count,
                callee_family_count=len(role_candidate.role_method_counts),
            ),
        )


declare_candidate_rule_detector(
    FacadeOnlyNominalAuthorityCandidate,
    high_confidence_spec(
        PatternId.NOMINAL_INTERFACE_WITNESS,
        "Nominal authority facade must own behavior, not only forward",
        "A nominal authority class whose public methods only call private module functions is a facade-only refactor. It hides bare helpers without moving semantic ownership, so the abstraction does not pay rent until the class owns the shared algorithm, policy state, registration axis, or derived projections.",
        "authority object with real semantic ownership rather than one-line private-helper delegation",
        "nominal authority methods are all pass-through calls to private functions",
        _AUTHORITATIVE_NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS,
        _CLASS_FAMILY_METHOD_ROLE_PARTIAL_VIEW_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}` exposes facade-only methods {candidate.method_names} "
        f"that delegate to private helpers {candidate.delegate_names}; move the algorithms into "
        "the authority or delete the facade."
    ),
    scaffold=lambda candidate: (
        f"class {candidate.class_name}:\n"
        "    # Method bodies own the algorithm/policy directly.\n"
        "    # Private helpers remain only for tiny local residue.\n"
        "    ..."
    ),
    codemod_patch=lambda candidate: (
        f"# Inline private delegate bodies {candidate.delegate_names} into `{candidate.class_name}` methods, "
        "then delete the private delegate functions.\n"
        "# If the class still has no state, registry, policy, or shared algorithm after inlining, delete the class instead."
    ),
    metrics=lambda candidate: OrchestrationMetrics(
        function_line_count=candidate.line_count,
        branch_site_count=0,
        call_site_count=len(candidate.delegate_names),
        parameter_count=len(candidate.method_names),
        callee_family_count=1,
    ),
    candidate_collector=_facade_only_nominal_authority_candidates,
    detector_name="FacadeOnlyNominalAuthorityDetector",
)


def _alias_only_authority_certificate(
    candidate: AliasOnlyNominalAuthorityCandidate,
) -> CompressionCertificate:
    alias_count = len(candidate.alias_names)
    return CompressionCertificate.from_object_family(
        manual_object_count=alias_count,
        replacement_shape=ObjectFamilyShape(shared_objects=("authority_shell",)),
        semantic_axes=candidate.alias_names,
        residual_object_count=alias_count,
        wiring_object_count=len(candidate.delegate_names),
    )


def _module_authority_reexport_certificate(
    candidate: ModuleAuthorityReexportCatalogCandidate,
) -> CompressionCertificate:
    alias_count = len(candidate.alias_names)
    return CompressionCertificate.from_object_family(
        manual_object_count=alias_count,
        replacement_shape=ObjectFamilyShape(
            shared_objects=("authority_reexport_catalog",)
        ),
        semantic_axes=candidate.alias_names,
        residual_object_count=alias_count,
        wiring_object_count=len(candidate.delegate_names),
    )


declare_candidate_rule_detector(
    AliasOnlyNominalAuthorityCandidate,
    high_confidence_spec(
        PatternId.NOMINAL_INTERFACE_WITNESS,
        "Nominal authority alias catalog does not pay rent",
        "A nominal authority whose public surface is only aliases to private helpers is name shuffling, not semantic compression. The authority must own a shared algorithm, policy state, registry axis, or derived projection; otherwise keep the helpers local until a real invariant is found.",
        "authority object with owned shared behavior rather than alias catalog",
        "nominal authority class contains only public aliases to helper functions",
        _AUTHORITATIVE_NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS,
        _CLASS_FAMILY_METHOD_ROLE_PARTIAL_VIEW_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}` aliases public names {candidate.alias_names} "
        f"to delegates {candidate.delegate_names}; this is not a rent-paying authority. "
        f"Rent proof: {_alias_only_authority_certificate(candidate).rent_proof_summary}."
    ),
    scaffold=lambda candidate: (
        f"class {candidate.class_name}:\n"
        "    # Own the common algorithm here, or delete the authority.\n"
        "    ..."
    ),
    codemod_patch=lambda candidate: (
        f"# Delete alias-only authority `{candidate.class_name}` or replace it with "
        "a real shared algorithm/policy object. Do not re-export bound aliases as a refactor.\n"
        f"# Rent proof: {_alias_only_authority_certificate(candidate).rent_proof_summary}"
    ),
    compression_certificate=_alias_only_authority_certificate,
    metrics=lambda candidate: OrchestrationMetrics(
        function_line_count=candidate.line_count,
        branch_site_count=0,
        call_site_count=len(candidate.delegate_names),
        parameter_count=len(candidate.alias_names),
        callee_family_count=1,
    ),
    candidate_collector=_alias_only_nominal_authority_candidates,
    detector_name="AliasOnlyNominalAuthorityDetector",
)


declare_candidate_rule_detector(
    ModuleAuthorityReexportCatalogCandidate,
    high_confidence_spec(
        PatternId.NOMINAL_INTERFACE_WITNESS,
        "Module authority re-export catalog hides non-paying refactor",
        "A module-level block that re-exports authority methods under the old helper names preserves the public helper surface while adding an object indirection. That is not semantic compression unless the authority removes duplicate algorithms or derives the surface from one invariant.",
        "authority-owned algorithm without compatibility re-export catalog",
        "module re-exports several authority methods as top-level helper aliases",
        _AUTHORITATIVE_NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS,
        _CLASS_FAMILY_METHOD_ROLE_PARTIAL_VIEW_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: (
        f"`{candidate.authority_name}` is re-exported through helper aliases "
        f"{candidate.alias_names}; remove the alias catalog or prove the authority owns a shared invariant. "
        f"Rent proof: {_module_authority_reexport_certificate(candidate).rent_proof_summary}."
    ),
    scaffold=lambda candidate: (
        f"class {candidate.authority_name.title().replace('_', '')}:\n"
        "    # One shared algorithm/projection schema, not one alias per helper.\n"
        "    ..."
    ),
    codemod_patch=lambda candidate: (
        f"# Delete module-level re-export aliases for `{candidate.authority_name}`.\n"
        "# Either call a real authority object directly at consumers, or collapse the helpers into one derived algebra.\n"
        f"# Rent proof: {_module_authority_reexport_certificate(candidate).rent_proof_summary}"
    ),
    compression_certificate=_module_authority_reexport_certificate,
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(candidate.alias_names),
        mapping_name=candidate.authority_name,
        field_names=candidate.alias_names,
    ),
    candidate_collector=_module_authority_reexport_catalog_candidates,
    detector_name="ModuleAuthorityReexportCatalogDetector",
)


declare_candidate_rule_detector(
    CollectionAuthorityStreamAlgebraCandidate,
    high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Collection authority should derive from one stream algebra",
        "A collection authority with several methods that repeat source selection, projection, and tuple/sorted materialization is still manually declaring a product of stream mechanics. The semantic normal form is one typed stream/spec algebra whose methods only declare the source stream and projection.",
        "typed candidate stream algebra deriving projection and materialization mechanics",
        "collection authority repeats stream projection and materialization across methods",
        _AUTHORITATIVE_SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}` repeats stream materialization across methods "
        f"{candidate.method_names}; derive them from one CandidateStream/CandidateProjection algebra."
    ),
    scaffold=lambda candidate: (
        "@dataclass(frozen=True)\n"
        "class CandidateStream(Generic[T]):\n"
        "    items: Iterable[T]\n"
        "    sort_key: Callable[[T], object] | None = None\n"
        "    def materialized(self) -> tuple[T, ...]: ...\n\n"
        "class CandidateCollectionAuthority:\n"
        "    def named_function_candidates(...):\n"
        "        return CandidateStream(projected, sort_key).materialized()"
    ),
    codemod_patch=lambda candidate: (
        "# Extract the repeated projection/materialization tail into `CandidateStream.materialized()`.\n"
        "# Keep source-stream selection and projector invocation as the only method-specific residue."
    ),
    metrics=lambda candidate: OrchestrationMetrics(
        function_line_count=candidate.line_count,
        branch_site_count=0,
        call_site_count=len(candidate.method_names),
        parameter_count=len(candidate.method_names),
        callee_family_count=1,
    ),
    candidate_collector=_collection_authority_stream_algebra_candidates,
    detector_name="CollectionAuthorityStreamAlgebraDetector",
)


_PREDICATE_GRAMMAR_AUTHORITY_SUFFIXES = (
    "Authority",
    "Builder",
    "Catalog",
    "Decoder",
    "Extractor",
    "Pipeline",
    "Profile",
    "Projection",
    "Renderer",
)


class AstTypeIsinstanceNameProjection:
    def from_expr(self, node: ast.AST) -> str | None:
        attribute = as_ast(node, ast.Attribute)
        if attribute is not None and name_id(attribute.value) == "ast":
            return attribute.attr
        name = name_id(node)
        return name if name is not None and name[:1].isupper() else None

    def from_isinstance_call(self, call: ast.Call) -> tuple[str, ...]:
        if _call_name(call.func) != "isinstance" or len(call.args) < 2:
            return ()
        type_expr = call.args[1]
        if isinstance(type_expr, ast.Tuple):
            return sorted_tuple(
                (
                    type_name
                    for element in type_expr.elts
                    if (type_name := self.from_expr(element)) is not None
                )
            )
        type_name = self.from_expr(type_expr)
        return () if type_name is None else (type_name,)


AST_TYPE_ISINSTANCE_NAME_PROJECTION = AstTypeIsinstanceNameProjection()


def _uses_ast_traversal(node: ast.AST) -> bool:
    return any(
        (
            isinstance(call, ast.Call)
            and _call_name(call.func) in {"_walk_nodes", "ast.walk"}
            for call in _walk_nodes(node)
        )
    )


def _predicate_grammar_score(method: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    return sum(
        (
            isinstance(node, (ast.If, ast.BoolOp, ast.Compare))
            or (
                isinstance(node, ast.Call)
                and bool(AST_TYPE_ISINSTANCE_NAME_PROJECTION.from_isinstance_call(node))
            )
        )
        for node in _walk_nodes(method)
    )


def _inline_ast_predicate_grammar_candidates(
    module: ParsedModule,
) -> tuple[InlineAstPredicateGrammarCandidate, ...]:
    candidates: list[InlineAstPredicateGrammarCandidate] = []
    for node in module.module.body:
        if not isinstance(node, ast.ClassDef) or not node.name.endswith(
            _PREDICATE_GRAMMAR_AUTHORITY_SUFFIXES
        ):
            continue
        for method in CLASS_NODE_AUTHORITY.methods(node):
            ast_type_names = sorted_tuple(
                {
                    type_name
                    for call in _walk_nodes(method)
                    if isinstance(call, ast.Call)
                    for type_name in (
                        AST_TYPE_ISINSTANCE_NAME_PROJECTION.from_isinstance_call(call)
                    )
                }
            )
            predicate_count = _predicate_grammar_score(method)
            traversal_count = sum(
                (isinstance(loop, (ast.For, ast.While)) for loop in _walk_nodes(method))
            )
            if (
                predicate_count < 6
                or traversal_count == 0
                or not ast_type_names
                or not _uses_ast_traversal(method)
            ):
                continue
            candidates.append(
                InlineAstPredicateGrammarCandidate(
                    file_path=str(module.path),
                    line=method.lineno,
                    class_name=node.name,
                    method_name=method.name,
                    ast_type_names=ast_type_names,
                    predicate_count=predicate_count,
                    traversal_count=traversal_count,
                    line_count=max(
                        1, (method.end_lineno or method.lineno) - method.lineno + 1
                    ),
                )
            )
    return sorted_tuple(
        candidates, key=lambda item: (item.file_path, item.line, item.class_name)
    )


def _inline_ast_predicate_grammar_certificate(
    candidate: InlineAstPredicateGrammarCandidate,
) -> CompressionCertificate:
    return CompressionCertificate.from_object_family(
        manual_object_count=candidate.predicate_count + candidate.traversal_count,
        replacement_shape=ObjectFamilyShape(
            shared_objects=("ast_predicate_grammar", "matcher_runner"),
            per_axis_objects=("node_type_rule",),
        ),
        semantic_axes=candidate.ast_type_names,
        residual_object_count=max(1, len(candidate.ast_type_names)),
    )


declare_candidate_rule_detector(
    InlineAstPredicateGrammarCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Authority method contains inline AST predicate grammar",
        "A nominal authority method that still hand-codes AST traversal, isinstance checks, attribute guards, and boolean predicate ladders has only moved the smell. The deeper normal form is a declarative matcher/effect-step grammar: node types and field predicates are data, while traversal and failure semantics live in one reusable ABC.",
        "declarative AST matcher grammar with traversal and predicate semantics owned once",
        "authority method repeats AST traversal and predicate mechanics inline",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS,
        _NORMALIZED_AST_CLASS_FAMILY_METHOD_ROLE_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}.{candidate.method_name}` has "
        f"{candidate.predicate_count} inline AST predicates over {candidate.ast_type_names} "
        f"inside {candidate.traversal_count} traversal block(s); move this into a matcher grammar."
    ),
    scaffold=lambda candidate: (
        "class AstPredicateRule(ABC):\n"
        "    node_types: ClassVar[tuple[type[ast.AST], ...]]\n"
        "    def matches(self, node: ast.AST) -> bool: ...\n\n"
        "class AstPredicateGrammar(ABC):\n"
        "    rules: ClassVar[tuple[AstPredicateRule, ...]]\n"
        "    def matches_anywhere(self, root: ast.AST): ..."
    ),
    codemod_patch=lambda candidate: (
        f"# Replace inline predicate ladder in `{candidate.class_name}.{candidate.method_name}` "
        "with declarative `AstPredicateRule` rows and one traversal runner.\n"
        "# Keep node type, field name, operator, and projection residue as typed rule data; "
        "do not hide repeated predicates inside another authority method."
    ),
    metrics=lambda candidate: OrchestrationMetrics(
        function_line_count=candidate.line_count,
        branch_site_count=candidate.predicate_count,
        call_site_count=candidate.traversal_count,
        parameter_count=len(candidate.ast_type_names),
        callee_family_count=1,
    ),
    compression_certificate=_inline_ast_predicate_grammar_certificate,
    candidate_collector=_inline_ast_predicate_grammar_candidates,
)


@dataclass(frozen=True)
class ProjectionPropertyFamilyCandidate(ClassLineWitnessCandidate):
    property_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    base_names: tuple[str, ...]

    @property
    def evidence_locations(self) -> tuple[SourceLocation, ...]:
        return tuple(
            (
                SourceLocation(
                    self.file_path, line, f"{self.class_name}.{property_name}"
                )
                for line, property_name in zip(
                    self.line_numbers, self.property_names, strict=True
                )
            )
        )


@dataclass(frozen=True)
class SelfAttributeAuthority:
    def attr_name(self, node: ast.AST) -> str | None:
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and (node.value.id == "self")
        ):
            return node.attr
        return None


SELF_ATTRIBUTE_AUTHORITY = SelfAttributeAuthority()


def _is_path_projection_part(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    if isinstance(node, ast.JoinedStr):
        return all(
            (
                isinstance(value, ast.Constant)
                or (
                    isinstance(value, ast.FormattedValue)
                    and SELF_ATTRIBUTE_AUTHORITY.attr_name(value.value) is not None
                )
                for value in node.values
            )
        )
    return False


def _path_projection_base(returned: ast.AST) -> str | None:
    node = returned
    saw_path_part = False
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        if not _is_path_projection_part(node.right):
            return None
        saw_path_part = True
        node = node.left
    if not saw_path_part:
        return None
    return SELF_ATTRIBUTE_AUTHORITY.attr_name(node)


def _projection_property_family_candidates(
    module: ParsedModule,
) -> tuple[ProjectionPropertyFamilyCandidate, ...]:
    candidates: list[ProjectionPropertyFamilyCandidate] = []
    for class_node in (
        node for node in _walk_nodes(module.module) if isinstance(node, ast.ClassDef)
    ):
        properties: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str]] = []
        for statement in class_node.body:
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not any(
                (
                    _ast_terminal_name(decorator) == "property"
                    for decorator in statement.decorator_list
                )
            ):
                continue
            body = _trim_docstring_body(statement.body)
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                continue
            base_name = _path_projection_base(body[0].value)
            if base_name is None:
                continue
            properties.append((statement, base_name))
        if len(properties) < 3:
            continue
        ordered = sorted_tuple(properties, key=lambda item: item[0].lineno)
        candidates.append(
            ProjectionPropertyFamilyCandidate(
                file_path=str(module.path),
                line=class_node.lineno,
                class_name=class_node.name,
                property_names=tuple((function.name for function, _ in ordered)),
                line_numbers=tuple((function.lineno for function, _ in ordered)),
                base_names=sorted_tuple({base_name for _, base_name in ordered}),
            )
        )
    return sorted_tuple(
        candidates, key=lambda item: (item.file_path, item.line, item.class_name)
    )


declare_candidate_rule_detector(
    ProjectionPropertyFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.DESCRIPTOR_DERIVED_VIEW,
        "Path projection properties should be derived descriptors",
        "Several properties project Path-valued views from owned base fields through the same `/` algebra. That is a descriptor-derived view family: the varying suffixes should be data while the projection algorithm lives in one reusable descriptor.",
        "single descriptor authority for repeated Path projection properties",
        "same class repeats Path projection properties over owned base fields",
        _AUTHORITATIVE_PROVENANCE_UNIT_RATE_COHERENCE_CAPABILITY_TAGS,
        _PROJECTION_HELPER_NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS,
    ),
    summary=lambda projection_candidate: f"`{projection_candidate.class_name}` repeats Path projection properties {', '.join(projection_candidate.property_names)} over bases {', '.join(projection_candidate.base_names)}.",
    evidence=lambda projection_candidate: projection_candidate.evidence_locations,
    scaffold=lambda projection_candidate: "@dataclass(frozen=True)\nclass PathProjection:\n    base_attr: str\n    parts: tuple[str, ...]\n    def __get__(self, instance, owner=None) -> Path: ...",
    codemod_patch=lambda projection_candidate: "# Replace repeated @property path projections with PathProjection descriptors.\n# Keep only base attribute and path parts as declarative data.",
    metrics=lambda projection_candidate: MappingMetrics(
        mapping_site_count=len(projection_candidate.property_names),
        field_count=len(projection_candidate.base_names),
        mapping_name=f"{projection_candidate.class_name} path projection",
        field_names=projection_candidate.property_names,
        source_name=", ".join(projection_candidate.base_names),
    ),
    candidate_collector=_projection_property_family_candidates,
)


def _collection_projection_property_shape(
    returned: ast.AST,
) -> tuple[str, str] | None:
    return (
        Maybe.of(as_ast(returned, ast.Call))
        .filter(
            lambda call: name_id(call.func) in {"tuple", "list", "set", "frozenset"}
            and len(call.args) == 1
            and not call.keywords
        )
        .project(lambda call: as_ast(call.args[0], ast.GeneratorExp))
        .filter(lambda generator: len(generator.generators) == 1)
        .map(lambda generator: (generator, generator.generators[0]))
        .filter(lambda context: not context[1].ifs)
        .combine(
            lambda context: SELF_ATTRIBUTE_AUTHORITY.attr_name(context[1].iter),
            lambda context, collection_name: (
                collection_name,
                as_ast(context[0].elt, ast.Attribute),
                name_id(context[1].target),
            ),
        )
        .project(
            lambda context: (
                (
                    context[0],
                    context[1].attr,
                )
                if context[1] is not None
                and context[2] is not None
                and name_id(context[1].value) == context[2]
                else None
            )
        )
        .unwrap_or_none()
    )


def _is_property_method(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    return any(
        (
            _ast_terminal_name(decorator) == "property"
            for decorator in method.decorator_list
        )
    )


def _collection_projection_property_family_candidates(
    module: ParsedModule,
) -> tuple[CollectionProjectionPropertyFamilyCandidate, ...]:
    candidates: list[CollectionProjectionPropertyFamilyCandidate] = []
    for class_node in (
        node for node in _walk_nodes(module.module) if isinstance(node, ast.ClassDef)
    ):
        properties: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str, str]] = []
        for statement in CLASS_NODE_AUTHORITY.methods(class_node):
            if not _is_property_method(statement):
                continue
            body = _trim_docstring_body(statement.body)
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                continue
            shape = _collection_projection_property_shape(body[0].value)
            if shape is None:
                continue
            collection_name, projected_attribute_name = shape
            properties.append((statement, collection_name, projected_attribute_name))
        grouped: dict[str, list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str]]] = (
            defaultdict(list)
        )
        for statement, collection_name, projected_attribute_name in properties:
            grouped[collection_name].append((statement, projected_attribute_name))
        for collection_name, grouped_properties in grouped.items():
            if len(grouped_properties) < 2:
                continue
            ordered = sorted_tuple(grouped_properties, key=lambda item: item[0].lineno)
            candidates.append(
                CollectionProjectionPropertyFamilyCandidate(
                    file_path=str(module.path),
                    line=class_node.lineno,
                    class_name=class_node.name,
                    property_names=tuple((statement.name for statement, _ in ordered)),
                    line_numbers=tuple((statement.lineno for statement, _ in ordered)),
                    collection_name=collection_name,
                    projected_attribute_names=tuple(
                        (attribute_name for _, attribute_name in ordered)
                    ),
                    line_count=(class_node.end_lineno or class_node.lineno)
                    - class_node.lineno
                    + 1,
                )
            )
    return sorted_tuple(
        candidates, key=lambda item: (item.file_path, item.line, item.class_name)
    )


declare_candidate_rule_detector(
    CollectionProjectionPropertyFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.DESCRIPTOR_DERIVED_VIEW,
        "Collection projection properties should be derived descriptors",
        "Sibling properties that only map one owned collection to member attributes are descriptor-derived views. Repeating `tuple(item.attr for item in self.collection)` per property makes each projected attribute look like behavior when the actual semantic object is the collection projection relation.",
        "single collection-projection descriptor parameterized by collection and member attribute",
        "same class repeats collection projection properties over one owned collection",
        _AUTHORITATIVE_PROVENANCE_UNIT_RATE_COHERENCE_CAPABILITY_TAGS,
        _PROJECTION_HELPER_NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}` repeats collection projection properties "
        f"{candidate.property_names} over `self.{candidate.collection_name}` "
        f"for member attributes {candidate.projected_attribute_names}."
    ),
    evidence=lambda candidate: candidate.evidence_locations,
    scaffold=lambda candidate: (
        "@dataclass(frozen=True)\n"
        "class CollectionAttributeProjection:\n"
        "    collection_attr: str\n"
        "    member_attr: str\n"
        "    def __get__(self, instance, owner=None): ..."
    ),
    codemod_patch=lambda candidate: (
        "# Replace repeated collection projection @property methods with one "
        "CollectionAttributeProjection descriptor; keep only collection and "
        "member attribute names as class-level data."
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(candidate.property_names),
        mapping_name=f"{candidate.class_name}.{candidate.collection_name}",
        field_names=candidate.projected_attribute_names,
    ),
    candidate_collector=_collection_projection_property_family_candidates,
)


@dataclass(frozen=True)
class LiveTemplatePayloadFamilyCandidate(ClassLineWitnessCandidate):
    method_names: tuple[str, ...]
    line_numbers: tuple[int, ...]

    @property
    def evidence_locations(self) -> tuple[SourceLocation, ...]:
        return tuple(
            (
                SourceLocation(self.file_path, line, f"{self.class_name}.{method_name}")
                for line, method_name in zip(
                    self.line_numbers, self.method_names, strict=True
                )
            )
        )


def _returns_direct_text_template(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    body = _trim_docstring_body(function.body)
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    returned = body[0].value
    return isinstance(returned, ast.JoinedStr) or (
        isinstance(returned, ast.Constant) and isinstance(returned.value, str)
    )


def _live_template_payload_family_candidates(
    module: ParsedModule,
) -> tuple[LiveTemplatePayloadFamilyCandidate, ...]:
    candidates: list[LiveTemplatePayloadFamilyCandidate] = []
    for class_node in (
        node for node in _walk_nodes(module.module) if isinstance(node, ast.ClassDef)
    ):
        template_methods = tuple(
            (
                statement
                for statement in class_node.body
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                and _returns_direct_text_template(statement)
            )
        )
        if len(template_methods) < 3:
            continue
        ordered = sorted_tuple(template_methods, key=lambda item: item.lineno)
        candidates.append(
            LiveTemplatePayloadFamilyCandidate(
                file_path=str(module.path),
                line=class_node.lineno,
                class_name=class_node.name,
                method_names=tuple((method.name for method in ordered)),
                line_numbers=tuple((method.lineno for method in ordered)),
            )
        )
    return sorted_tuple(
        candidates, key=lambda item: (item.file_path, item.line, item.class_name)
    )


declare_candidate_rule_detector(
    LiveTemplatePayloadFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Live template payload methods should be derived from template specs",
        "Several live methods return template payloads directly. Unlike dead payload emitters, these methods are real API, so the correct collapse is to keep the template declarations as data and derive the method surface from one generic template descriptor.",
        "single template-method descriptor authority for live text payload families",
        "same class repeats direct text-template return methods",
        _AUTHORITATIVE_PROVENANCE_UNIT_RATE_COHERENCE_CAPABILITY_TAGS,
        _EXPORT_NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS,
    ),
    summary=lambda template_candidate: f"`{template_candidate.class_name}` exposes live template payload methods {', '.join(template_candidate.method_names)}.",
    evidence=lambda template_candidate: template_candidate.evidence_locations,
    scaffold=lambda template_candidate: "@dataclass(frozen=True)\nclass TextTemplateMethod:\n    parameters: tuple[str, ...]\n    template: str\n    def __get__(self, instance, owner=None): ...",
    codemod_patch=lambda template_candidate: "# Replace direct template-return methods with TextTemplateMethod descriptors.\n# Keep template bodies declarative; derive the bound method API generically.",
    metrics=lambda template_candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(template_candidate.method_names),
        mapping_name=f"{template_candidate.class_name} templates",
        field_names=template_candidate.method_names,
    ),
    candidate_collector=_live_template_payload_family_candidates,
)


class PrivateCohortShouldBeModuleDetector(
    ConfiguredModuleCollectorCandidateDetector[PrivateCohortShouldBeModuleCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.STAGED_ORCHESTRATION,
        "Private subsystem cohort wants its own module",
        "One module is carrying a tightly-coupled private subsystem cohort as if it were a whole package. The architecture wants a dedicated module for that bounded context, with the original file reduced to orchestration or public entry points.",
        "explicit module-level subsystem boundaries with extracted private cohorts",
        "one file contains a dense private context/result/helper family that should move together",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS,
    )

    def _finding_for_candidate(
        self, cohort: PrivateCohortShouldBeModuleCandidate
    ) -> RefactorFinding:
        shared_tokens = ", ".join(cohort.shared_tokens[:3]) or "subsystem"
        sample_symbols = ", ".join(
            (
                symbol.symbol
                for symbol in sorted(
                    cohort.symbols,
                    key=lambda item: (-item.line_count, item.line, item.symbol),
                )[:3]
            )
        )
        target_module = _suggest_private_cohort_module_name(cohort)
        return self.build_finding(
            (
                f"`{cohort.module_name}` carries a private {shared_tokens} cohort across "
                f"{len(cohort.symbols)} top-level symbols / {cohort.total_cohort_lines} lines "
                f"inside a {cohort.module_line_count}-line module; extract `{sample_symbols}` "
                f"into a dedicated `{target_module}.py` module."
            ),
            cohort.evidence,
            scaffold=(
                f"# {target_module}.py\n@dataclass(frozen=True)\nclass {_camel_case('_'.join(cohort.shared_tokens[:2]) or 'subsystem')}Context:\n    ...\n\ndef run_{'_'.join(cohort.shared_tokens[:2]) or 'subsystem'}(...):\n    ...\n\n# Move the private context/result carriers and worker helpers here.\n# Leave only public orchestration entry points in the original module."
            ),
            codemod_patch=(
                f"# Extract the private {shared_tokens} cohort into `{target_module}.py`.\n# Move the cohort's private dataclasses, helper functions, and result carriers together.\n# Import the extracted helpers back into the original module only where public entry points still need them.\n# Keep sequencing, public APIs, and thin phase boundaries in the original file."
            ),
        )


class ParameterThreadFamilyDetector(
    ConfiguredModuleCollectorCandidateDetector[ParameterThreadFamilyCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_CONTEXT,
        "Repeated threaded semantic parameter family",
        "Several helpers keep re-threading the same semantic parameter bundle instead of carrying one nominal context. That weakens provenance and makes each helper signature a partially duplicated view of the same authority.",
        "one authoritative context/request record for a shared semantic parameter family",
        "the same semantic parameter bundle is threaded through several sibling helpers",
        _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
    )

    def _finding_for_candidate(
        self, parameter_family: ParameterThreadFamilyCandidate
    ) -> RefactorFinding:
        function_names = tuple((item.qualname for item in parameter_family.functions))
        return self.build_finding(
            f"Functions {', '.join(function_names[:4])} thread the same semantic parameter family `{', '.join(parameter_family.shared_parameter_names)}` across {len(parameter_family.functions)} helpers.",
            tuple((item.evidence for item in parameter_family.functions[:6])),
            scaffold=_authoritative_context_scaffold(parameter_family),
            codemod_patch=_authoritative_context_patch(parameter_family),
            metrics=ParameterThreadMetrics(
                function_count=len(parameter_family.functions),
                shared_parameter_count=len(parameter_family.shared_parameter_names),
                shared_parameter_names=parameter_family.shared_parameter_names,
            ),
        )


class SuffixAxisCompatibilitySurfaceDetector(
    ConfiguredModuleCollectorCandidateDetector[SuffixAxisSurfaceCandidate]
):
    candidate_collector = _suffix_axis_surface_candidates
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_CONTEXT,
        "Mirrored suffix-axis APIs should collapse to one authoritative context",
        "Several operations are exposed once per suffix-named axis, such as `*_for_context` and `*_for_session`. When the same axis split repeats across an owner, the code is usually maintaining adapter surfaces instead of choosing one authoritative request/context record and deriving any compatibility projection at the boundary.",
        "single authoritative context/request record instead of repeated suffix-axis adapter surfaces",
        "same owner repeats an operation family across the same suffix-named axes",
        _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _METHOD_ROLE_PARTIAL_VIEW_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, surface_candidate: SuffixAxisSurfaceCandidate
    ) -> RefactorFinding:
        axis_summary = " / ".join(surface_candidate.axis_names)
        operation_summary = ", ".join(surface_candidate.operation_names[:5])
        method_names = tuple(method.qualname for method in surface_candidate.methods)
        return self.build_finding(
            (
                f"`{surface_candidate.owner_name}` repeats suffix-axis APIs for axes {axis_summary} "
                f"across operations {operation_summary}."
            ),
            surface_candidate.evidence,
            scaffold=(
                "@dataclass(frozen=True)\nclass OperationContext:\n    ...\n\n# Route operations through one authoritative context/session/request record.\n# Keep at most one boundary adapter that constructs the authority, not one adapter per operation."
            ),
            codemod_patch=(
                f"# Collapse suffix-axis method family {method_names[:8]} onto one authoritative record.\n"
                "# Prefer one conversion point from the secondary axis into the primary axis, then delete per-operation mirrored wrappers."
            ),
            metrics=ParameterThreadMetrics(
                function_count=len(surface_candidate.operation_names),
                shared_parameter_count=len(surface_candidate.axis_names),
                shared_parameter_names=surface_candidate.axis_names,
            ),
        )


class SiblingRoleHelperSymmetryDetector(
    ModuleCollectorCandidateDetector[SiblingRoleHelperSymmetryCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.LOCAL_VALUE_AUTHORITY,
        "Sibling role helpers should collapse to one local authority",
        "One owner has private helpers whose names differ by a role token but whose control skeletons and parameters are parallel. That is usually one local computation split into symmetrical role-specific helpers, which makes future changes require duplicated edits.",
        "one authoritative local computation instead of parallel role-specific helpers",
        "same owner has role-token sibling helpers with matching control skeletons",
        _AUTHORITATIVE_SHARED_ALGORITHM_AUTHORITY_PROVENANCE_CAPABILITY_TAGS,
        _METHOD_ROLE_NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, helper_candidate: SiblingRoleHelperSymmetryCandidate
    ) -> RefactorFinding:
        helper_summary = ", ".join(helper_candidate.method_names)
        role_summary = " / ".join(helper_candidate.role_tokens)
        shared_summary = "_".join(helper_candidate.shared_tokens)
        return self.build_finding(
            (
                f"`{helper_candidate.owner_name}` splits `{shared_summary}` across role helpers "
                f"{helper_summary} for roles {role_summary}."
            ),
            helper_candidate.evidence,
            scaffold=(
                f"def resolve_{shared_summary}(...):\n    # Compute the role-specific values together while the branch facts are live.\n    ...\n    return left_value, right_value\n\n# Use a small record only if this result crosses a boundary; keep local-only pairs as values."
            ),
            codemod_patch=(
                f"# Collapse sibling helpers {helper_candidate.method_names} into one local authority.\n"
                "# Preserve role names at the assignment site instead of maintaining parallel helper bodies."
            ),
            metrics=ParameterThreadMetrics(
                function_count=len(helper_candidate.methods),
                shared_parameter_count=len(helper_candidate.shared_tokens),
                shared_parameter_names=helper_candidate.shared_tokens,
            ),
        )


declare_candidate_rule_detector(
    EnumStrategyDispatchCandidate,
    high_confidence_spec(
        PatternId.NOMINAL_STRATEGY_FAMILY,
        "Enum strategy ladder wants nominal family",
        "A closed enum/member dispatch ladder is choosing among behavior implementations inline. That wants an ABC-backed strategy family so each implementation guarantees one common method and the caller stops branching.",
        "nominal strategy family with one guaranteed call surface",
        "one owner branches over a closed enum/member family instead of delegating to implementation classes",
        _CLOSED_FAMILY_DISPATCH_NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_CAPABILITY_TAGS,
    ),
    summary=lambda dispatch_candidate: f"`{dispatch_candidate.qualname}` branches on `{dispatch_candidate.dispatch_axis}` across closed cases {', '.join(dispatch_candidate.case_names)} and should delegate to a nominal strategy family.",
    scaffold=lambda dispatch_candidate: _nominal_strategy_scaffold(dispatch_candidate),
    codemod_patch=lambda dispatch_candidate: _nominal_strategy_patch(
        dispatch_candidate
    ),
    metrics=lambda dispatch_candidate: DispatchCountMetrics.from_literal_family(
        dispatch_axis=dispatch_candidate.dispatch_axis,
        literal_cases=dispatch_candidate.case_names,
    ),
    candidate_collector=ENUM_DISPATCH_EXTRACTOR.strategy_candidates,
)


class ResidualClosedAxisIndirectionDetector(
    ModuleCollectorCandidateDetector[ResidualClosedAxisIndirectionCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_STRATEGY_FAMILY,
        "Enum-keyed table with residual branching should become a nominal strategy family",
        "A function that indexes an enum-keyed table and still branches on the same enum axis is not using the table as an authority. The table is a degenerate projection over behavior that still lives in branches. The stronger normal form is an ABC-backed strategy family keyed by the enum, with `AutoRegisterMeta` owning import-time registration and any table-like views derived from the family.",
        "metaclass-registry-backed nominal strategy family instead of enum table plus residual branching",
        "same function indexes an enum-keyed table and branches on that enum axis",
        _AUTHORITATIVE_DISPATCH_CLOSED_FAMILY_DISPATCH_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _PROJECTION_DICT_BRANCH_DISPATCH_CLOSED_FAMILY_CASES_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, axis_candidate: ResidualClosedAxisIndirectionCandidate
    ) -> RefactorFinding:
        residual_cases = ", ".join(axis_candidate.residual_case_names)
        table_cases = ", ".join(axis_candidate.table_case_names)
        value_summary = ", ".join(axis_candidate.table_value_summaries[:4])
        return self.build_finding(
            (
                f"`{axis_candidate.qualname}` indexes `{axis_candidate.table_name}` by "
                f"`{axis_candidate.axis_expression}` for `{axis_candidate.enum_name}` cases {table_cases}, "
                f"but still branches on residual cases {residual_cases}."
            ),
            axis_candidate.evidence,
            scaffold=(
                f'from abc import ABC, abstractmethod\nfrom typing import ClassVar\nfrom metaclass_registry import AutoRegisterMeta\n\nclass AxisPolicy(ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = "axis_key"\n    __skip_if_no_key__ = True\n    axis_key: ClassVar[{axis_candidate.enum_name}]\n\n    @classmethod\n    def for_key(cls, key: {axis_candidate.enum_name}):\n        return cls.__registry__[key]()\n\n    @abstractmethod\n    def project(self, source): ...\n\n    @abstractmethod\n    def run(self, ctx): ...\n\n# Move the table projection and residual branch behavior into enum-keyed policy subclasses.\n# Derive table-like views from AxisPolicy.__registry__ only if callers still need them.'
            ),
            codemod_patch=(
                f"# Replace `{axis_candidate.table_name}[{axis_candidate.axis_expression}]` plus residual "
                f"`{axis_candidate.enum_name}` branching with `AxisPolicy.for_key({axis_candidate.axis_expression})`.\n"
                f"# Move projections ({value_summary}) and per-case behavior into registered `AxisPolicy` subclasses."
            ),
            metrics=DispatchCountMetrics.from_literal_family(
                dispatch_axis=axis_candidate.enum_name,
                literal_cases=axis_candidate.table_case_names,
            ),
        )


class RepeatedEnumStrategyDispatchDetector(
    ModuleCollectorCandidateDetector[RepeatedEnumStrategyDispatchCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_STRATEGY_FAMILY,
        "Repeated closed-strategy dispatch should centralize in one nominal strategy family",
        "Several owners re-dispatch the same closed enum family inline. The docs treat that as duplicated strategy orchestration: dispatch should happen once through one authoritative nominal strategy family or one shared strategy substrate.",
        "single authoritative nominal strategy family for a repeated closed dispatch axis",
        "same closed enum family is re-dispatched across sibling functions or methods",
        _CLOSED_FAMILY_DISPATCH_AUTHORITATIVE_DISPATCH_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS,
    )

    def _finding_for_candidate(
        self, dispatch_candidate: RepeatedEnumStrategyDispatchCandidate
    ) -> RefactorFinding:
        evidence = tuple((item.evidence for item in dispatch_candidate.functions[:6]))
        representative = dispatch_candidate.functions[0]
        function_names = ", ".join(
            (item.qualname for item in dispatch_candidate.functions[:4])
        )
        return self.build_finding(
            f"Functions {function_names} each re-dispatch `{dispatch_candidate.enum_family}` cases {', '.join(dispatch_candidate.shared_case_names)} inline.",
            evidence,
            scaffold=_nominal_strategy_scaffold(representative),
            codemod_patch=_nominal_strategy_patch(representative),
            metrics=DispatchCountMetrics(
                dispatch_site_count=len(dispatch_candidate.functions),
                dispatch_axis=dispatch_candidate.enum_family,
                literal_cases=dispatch_candidate.shared_case_names,
            ),
        )


class InlineEnumSubsetGuardDetector(
    ModuleCollectorCandidateDetector[InlineEnumSubsetGuardCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Inline enum subset guard should derive from enum-owned policy",
        "A branch that hardcodes an enum-member subset is a closed-axis policy table in disguise. The policy should be owned by the enum member or a typed row family, with any lookup derived exhaustively from that type-safe source.",
        "type-safe enum-owned policy instead of inline enum subset literals",
        "function branches on a hand-enumerated subset of one closed enum axis",
        _CLOSED_FAMILY_DISPATCH_AUTHORITATIVE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _BRANCH_DISPATCH_PROJECTION_DICT_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, guard_candidate: InlineEnumSubsetGuardCandidate
    ) -> RefactorFinding:
        cases = ", ".join(
            (
                f"{guard_candidate.enum_name}.{case_name}"
                for case_name in guard_candidate.case_names
            )
        )
        return self.build_finding(
            (
                f"`{guard_candidate.function_name}` checks `{guard_candidate.axis_expression} "
                f"{guard_candidate.operator} {{{cases}}}`; move that subset into enum-owned typed policy."
            ),
            (guard_candidate.evidence,),
            scaffold=(
                f"@dataclass(frozen=True)\nclass PolicyRow:\n    key: {guard_candidate.enum_name}\n    requires_policy: bool\n\ndef exhaustive_enum_lookup(enum_type, rows):\n    by_key = {{row.key: row for row in rows}}\n    if set(by_key) != set(enum_type):\n        raise TypeError('incomplete enum policy')\n    return by_key\n\nPOLICY_BY_KEY = exhaustive_enum_lookup(...)\nif POLICY_BY_KEY[{guard_candidate.axis_expression}].requires_policy:\n    ..."
            ),
            codemod_patch=(
                f"# Replace inline subset {{{cases}}} with a policy owned by `{guard_candidate.enum_name}`.\n"
                "# Derive any enum-keyed dict from enum members or typed policy rows, and fail if coverage is incomplete."
            ),
            metrics=DispatchCountMetrics.from_literal_family(
                dispatch_axis=guard_candidate.enum_name,
                literal_cases=guard_candidate.case_names,
            ),
        )


class SplitDispatchAuthorityDetector(
    ModuleCollectorCandidateDetector[SplitDispatchAuthorityCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_STRATEGY_FAMILY,
        "Cooperating dispatch layers should collapse into one product-family authority",
        "The docs treat repeated cooperating dispatch layers as split authority. When one orchestration function selects a strategy-family implementation and separately routes another axis through `singledispatch`, the operation usually wants one authoritative product-family policy or one request-dispatched plan.",
        "single authoritative product-family or request-dispatched policy for cooperating dispatch axes",
        "one orchestrator combines a strategy-family selector with a separate singledispatch generic",
        _AUTHORITATIVE_DISPATCH_NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS,
        _CLASS_FAMILY_FACTORY_DISPATCH_REPEATED_METHOD_ROLES_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, dispatch_candidate: SplitDispatchAuthorityCandidate
    ) -> RefactorFinding:
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
        return self.build_finding(
            (
                f"`{dispatch_candidate.qualname}` combines strategy selector "
                f"`{dispatch_candidate.strategy_root_name}.{dispatch_candidate.selector_method_name}({dispatch_candidate.strategy_axis_expression})` "
                f"with singledispatch `{dispatch_candidate.generic_function_name}({dispatch_candidate.generic_axis_expression})` "
                f"through callback `{dispatch_candidate.bridge_callback_name}`, splitting one operation across two dispatch authorities."
            ),
            evidence,
            scaffold=(
                "@dataclass(frozen=True)\nclass DispatchPlan:\n    strategy: object\n    source_type: type[object]\n\nclass ProductPolicy(ABC):\n    plan_key: ClassVar[DispatchPlan]\n    def run(self, request): ...\n"
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


class EmptyLeafProductFamilyDetector(
    ModuleCollectorCandidateDetector[EmptyLeafProductFamilyCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "Empty multiple-inheritance leaves should collapse into one product-family authority",
        "The docs allow mixins for orthogonal reusable concerns, but empty leaf classes that merely enumerate all combinations of two reusable axes are usually a handwritten product table in inheritance form. That product should become one keyed authority or one product-family selector.",
        "single authoritative keyed product family instead of empty inheritance combinations",
        "empty leaf classes encode the full Cartesian product of two reusable inheritance axes",
        _AUTHORITATIVE_DISPATCH_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS,
        _CLASS_FAMILY_REPEATED_METHOD_ROLES_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, product_candidate: EmptyLeafProductFamilyCandidate
    ) -> RefactorFinding:
        left_axis = ", ".join(product_candidate.left_axis_base_names)
        right_axis = ", ".join(product_candidate.right_axis_base_names)
        leaf_preview = ", ".join(product_candidate.leaf_class_names[:6])
        return self.build_finding(
            (
                f"Empty leaf classes {leaf_preview} encode `{left_axis}` x `{right_axis}` through multiple inheritance instead of one product-family authority."
            ),
            product_candidate.evidence,
            scaffold=(
                "@dataclass(frozen=True)\nclass ProductRule:\n    axis_left: object\n    axis_right: object\n    policy_type: type[object]\n\nPRODUCT_RULES = (...)\n"
            ),
            codemod_patch=(
                "# Replace the empty Cartesian-product leaf classes with one keyed product table or one nominal selector family.\n# Keep only irreducible axis-local behavior on the reusable bases; do not encode the cross product as `pass` subclasses."
            ),
            metrics=DispatchCountMetrics.from_literal_family(
                dispatch_axis=(
                    f"{' | '.join(product_candidate.left_axis_base_names)} x {' | '.join(product_candidate.right_axis_base_names)}"
                ),
                literal_cases=product_candidate.leaf_class_names,
            ),
        )


class ClosedConstantSelectorDetector(
    ModuleCollectorCandidateDetector[ClosedConstantSelectorCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Closed selector over sibling constants should derive from one selector table",
        "The docs treat branch ladders that choose among sibling specs, plans, contracts, or other immutable constants as duplicated selector logic once the constant family already exists. The selector should collapse into one authoritative keyed table or selector record so wrappers and downstream views are derived.",
        "single authoritative selector table for a closed constant family",
        "one function branches over a small predicate family and returns sibling constants or one shared wrapper around them",
        _AUTHORITATIVE_CLOSED_FAMILY_DISPATCH_PROVENANCE_CAPABILITY_TAGS,
        _BUILDER_CALL_DATAFLOW_ROOT_PREDICATE_CHAIN_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, selector_candidate: ClosedConstantSelectorCandidate
    ) -> RefactorFinding:
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
        return self.build_finding(
            (
                f"`{selector_candidate.qualname}` branches over {guard_summary}, returning {wrapper_summary}"
                f"sibling constants {constants_preview} from `{family_label}`."
            ),
            selector_candidate.evidence,
            scaffold=(
                "@dataclass(frozen=True)\nclass SelectorRule:\n    key: object\n    selected: object\n\nSELECTOR_RULES = (\n    SelectorRule(key=..., selected=...),\n)\n_SELECTED_BY_KEY = {rule.key: rule.selected for rule in SELECTOR_RULES}\n"
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


class DerivedWrapperSpecShadowDetector(
    ModuleCollectorCandidateDetector[DerivedWrapperSpecShadowCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Generated wrapper spec family should collapse into the authoritative spec family",
        "The docs treat writable wrapper-spec tables as secondary authorities when they just point back at an existing spec family and feed code generation. Wrapper metadata should live on the authoritative spec records so generated wrappers are derived from one source rather than synchronized across parallel tables.",
        "single authoritative spec family carrying wrapper-generation metadata",
        "secondary spec table references an authoritative spec family entry-by-entry and is only consumed by wrapper generation",
        _AUTHORITATIVE_PROVENANCE_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS,
        _BUILDER_CALL_DATAFLOW_ROOT_SCOPED_SHAPE_WRAPPER_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, shadow_candidate: DerivedWrapperSpecShadowCandidate
    ) -> RefactorFinding:
        primary_family_label = (
            shadow_candidate.primary_family_name
            or shadow_candidate.primary_constructor_name
        )
        constant_preview = ", ".join(shadow_candidate.primary_constant_names[:4])
        builder_preview = ", ".join(shadow_candidate.builder_names[:3])
        return self.build_finding(
            (
                f"`{shadow_candidate.derived_family_name}` re-encodes wrapper metadata over authoritative family "
                f"`{primary_family_label}` through link field `{shadow_candidate.link_field_name}` for {constant_preview}, "
                f"then feeds generated wrappers via {builder_preview}."
            ),
            shadow_candidate.evidence,
            scaffold=(
                "@dataclass(frozen=True)\nclass ExecutionSpec:\n    key: object\n    runner: object\n    wrapper_name: str | None = None\n    wrapper_defaults: dict[str, object] = field(default_factory=dict)\n\ndef build_wrapper(spec: ExecutionSpec): ...\n"
            ),
            codemod_patch=(
                f"# Remove parallel family `{shadow_candidate.derived_family_name}`.\n# Move `{', '.join(shadow_candidate.extra_field_names) or 'wrapper metadata'}` onto the authoritative `{shadow_candidate.primary_constructor_name}` records and derive wrappers directly from that family."
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


declare_candidate_rule_detector(
    ModuleKeyedSelectionHelperCandidate,
    high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Local keyed-selection helper should collapse into the generic keyed-record table",
        "The docs push reusable table/index machinery into one authoritative substrate. When a module defines a local selection-rule dataclass, a dict-index builder, and a keyed lookup helper that power multiple rule tables, it is reintroducing a second keyed-table framework instead of reusing the generic keyed-record helper.",
        "single authoritative keyed-record table substrate reused across module-level selector tables",
        "module-local selection helper framework powers multiple keyed rule tables",
        _AUTHORITATIVE_CLOSED_FAMILY_DISPATCH_PROVENANCE_CAPABILITY_TAGS,
        _BUILDER_CALL_DATAFLOW_ROOT_CLASS_FAMILY_OBSERVATION_TAGS,
    ),
    summary=lambda helper_candidate: f"`{helper_candidate.rule_class_name}`, `{helper_candidate.helper_function_name}`, and `{helper_candidate.lookup_function_name}` implement a local keyed-selection substrate for {', '.join(helper_candidate.rule_table_names[:4])} and indexes {', '.join(helper_candidate.index_table_names[:4])}.",
    evidence=lambda helper_candidate: helper_candidate.evidence,
    scaffold=lambda helper_candidate: 'KeyT = TypeVar("KeyT")\nRecordT = TypeVar("RecordT")\n\n@dataclass(frozen=True)\nclass KeyedRecordTable(Generic[KeyT, RecordT]):\n    records: tuple[RecordT, ...]\n    key_of: Callable[[RecordT], KeyT]\n    def require(self, key: KeyT, *, missing_error=None) -> RecordT: ...\n',
    codemod_patch=lambda helper_candidate: f"# Remove local keyed-selection helper `{helper_candidate.rule_class_name}` / `{helper_candidate.helper_function_name}` / `{helper_candidate.lookup_function_name}`.\n# Re-express these rule tables through the shared KeyedRecordTable substrate.",
    metrics=lambda helper_candidate: MappingMetrics(
        mapping_site_count=len(helper_candidate.rule_table_names),
        field_count=1,
        mapping_name=helper_candidate.rule_class_name,
        field_names=(helper_candidate.selected_field_name,),
        source_name=helper_candidate.helper_function_name,
        identity_field_names=("key",),
    ),
    candidate_collector=_module_keyed_selection_helper_candidates,
)


declare_candidate_rule_detector(
    CrossModuleAxisShadowFamilyCandidate,
    high_confidence_spec(
        PatternId.NOMINAL_STRATEGY_FAMILY,
        "Cross-module shadow family should collapse into one axis authority",
        "The docs require one authoritative owner per closed semantic axis. When one module already owns an enum/keyed family nominally and another module reintroduces a second family over the same cases, the axis has split authority and local behavior should derive from the authoritative family instead.",
        "single authoritative closed-axis family reused across modules",
        "same keyed enum axis is modeled by an authoritative family in one module and a shadow selector family in another",
        _AUTHORITATIVE_DISPATCH_NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS,
        _CLASS_FAMILY_FACTORY_DISPATCH_DATAFLOW_ROOT_OBSERVATION_TAGS,
    ),
    summary=lambda shadow_candidate: f"Axis `{shadow_candidate.key_type_name}` is already owned by `{shadow_candidate.authoritative.family_name}` but re-encoded by `{shadow_candidate.shadow.family_name}.{shadow_candidate.selector_method_name}` across cases {', '.join(shadow_candidate.shared_case_names[:4])}.",
    evidence=lambda shadow_candidate: shadow_candidate.evidence,
    scaffold=lambda shadow_candidate: _axis_policy_registry_scaffold("invariant(self)")
    + f"\n\ndef run_with_axis(axis: {_AXIS_POLICY_KEY_TYPE_NAME}, ...):\n    policy = {_AXIS_POLICY_ROOT_NAME}.for_key(axis)\n    # derive local execution from authoritative policy facts\n",
    codemod_patch=lambda shadow_candidate: f"# Remove shadow family `{shadow_candidate.shadow.family_name}`.\n# Derive local behavior from authoritative family `{shadow_candidate.authoritative.family_name}` instead of re-owning axis `{shadow_candidate.key_type_name}`.",
    metrics=lambda shadow_candidate: DISPATCH_ALGEBRA_AUTHORITY.axis_dispatch_metrics(
        shadow_candidate.shared_case_names, shadow_candidate.key_type_name
    ),
    detector_base=CrossModuleCollectorCandidateDetector,
    candidate_collector=_cross_module_axis_shadow_family_candidates,
)


class ResidualClosedAxisBranchingDetector(
    CrossModuleCollectorCandidateDetector[ResidualClosedAxisBranchingCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "Manual closed-axis branching should derive from existing keyed authority",
        "The docs require one authoritative owner per closed enum/key axis. When a keyed nominal family already owns that axis, downstream `if`/`match` ladders over the same cases become residual shadow dispatch.",
        "behavior derived from authoritative keyed family rather than downstream enum branching",
        "function branches on an enum axis already owned by a keyed nominal family in another module",
        _AUTHORITATIVE_DISPATCH_CLOSED_FAMILY_DISPATCH_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _BRANCH_DISPATCH_CLASS_FAMILY_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, residual_candidate: ResidualClosedAxisBranchingCandidate
    ) -> RefactorFinding:
        authoritative_family_names = ", ".join(
            (
                family_name
                for family_name, _, _ in residual_candidate.authoritative_families[:4]
            )
        )
        return self.build_finding(
            (
                f"`{residual_candidate.qualname}` branches {residual_candidate.branch_site_count} time(s) on axis "
                f"`{residual_candidate.key_type_name}` across cases {', '.join(residual_candidate.case_names)}, "
                f"even though authoritative family `{authoritative_family_names}` already owns that axis."
            ),
            residual_candidate.evidence,
            scaffold=(
                _axis_policy_registry_scaffold("apply(self, context)")
                + f"\n\ndef run(context):\n    policy = {_AXIS_POLICY_ROOT_NAME}.for_key(context.axis)\n    return policy.apply(context)\n"
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


class ParallelKeyedAxisFamilyDetector(
    CrossModuleCollectorCandidateDetector[ParallelKeyedAxisFamilyCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_STRATEGY_FAMILY,
        "Parallel keyed families should collapse into one axis authority",
        "The docs require one authoritative nominal owner per closed semantic axis. When two modules each define a keyed family over the same enum/key cases, the axis has split ownership even if both sides are nominal.",
        "single cross-module keyed-axis authority with module-local adapters derived from it",
        "same keyed enum axis is modeled by multiple nominal families across modules",
        _AUTHORITATIVE_DISPATCH_AUTHORITATIVE_NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS,
        _CLASS_FAMILY_FACTORY_DISPATCH_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, family_candidate: ParallelKeyedAxisFamilyCandidate
    ) -> RefactorFinding:
        shared_cases = ", ".join(family_candidate.shared_case_names[:4])
        label_clause = ""
        if (
            family_candidate.left.family_label is not None
            and family_candidate.left.family_label
            == family_candidate.right.family_label
        ):
            label_clause = (
                f" Both declare family label `{family_candidate.left.family_label}`."
            )
        return self.build_finding(
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
                + "\n\n# Keep one authoritative keyed family and let secondary modules derive local adapters/specs from it."
            ),
            codemod_patch=(
                f"# Collapse `{family_candidate.left.family_name}` and `{family_candidate.right.family_name}` onto one authoritative keyed family.\n"
                "# Move the irreducible case-specific hooks to that family or to a single derived adapter table, not two parallel nominal roots."
            ),
            metrics=DISPATCH_ALGEBRA_AUTHORITY.axis_dispatch_metrics(
                family_candidate.shared_case_names,
                family_candidate.key_type_name,
            ),
        )


declare_candidate_rule_detector(
    ParallelKeyedTableAxisCandidate,
    high_confidence_spec(
        PatternId.NOMINAL_STRATEGY_FAMILY,
        "Parallel keyed tables should collapse into one auto-registered semantic family",
        "The docs require one authoritative owner per closed semantic axis. When multiple modules maintain keyed tables over the same cases, those tables are usually shadow registries for one semantic family. The stronger default normal form is an ABC plus `AutoRegisterMeta`, with table-like views derived from `Family.__registry__` only when callers still need projections.",
        "single AutoRegisterMeta-backed semantic family with derived module-local projections",
        "same closed enum/key axis is encoded by multiple keyed tables across modules",
        _AUTHORITATIVE_DISPATCH_NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS,
        _PROJECTION_DICT_DATAFLOW_ROOT_BUILDER_CALL_OBSERVATION_TAGS,
    ),
    summary=lambda table_candidate: f"Axis `{table_candidate.key_type_name}` is restated by `{table_candidate.left.table_name}` and `{table_candidate.right.table_name}` across cases {', '.join(table_candidate.shared_case_names[:4])}.",
    evidence=lambda table_candidate: table_candidate.evidence,
    scaffold=lambda table_candidate: _axis_policy_registry_scaffold("run(self, request)")
    + f"\n\ndef run_{table_candidate.key_type_name.lower()}(method, request):\n    return {_AXIS_POLICY_ROOT_NAME}.__registry__[method].run(request)\n\n# Derive table-like projections from {_AXIS_POLICY_ROOT_NAME}.__registry__ only if legacy callers need them.\n",
    codemod_patch=lambda table_candidate: f"# Collapse `{table_candidate.left.table_name}` and `{table_candidate.right.table_name}` onto one AutoRegisterMeta-backed semantic family.\n# Replace hardcoded keyed tables with registered subclasses and route behavior through `Family.__registry__[key].run(...)`.\n# Keep any table-like surface as a derived read-only projection from the registry, not as a writable authority.",
    metrics=lambda table_candidate: MappingMetrics(
        mapping_site_count=2,
        field_count=max(len(table_candidate.shared_case_names), 1),
        mapping_name=table_candidate.left.table_name,
        field_names=table_candidate.shared_case_names,
        source_name=table_candidate.key_type_name,
        identity_field_names=("key",),
    ),
    detector_base=CrossModuleCollectorCandidateDetector,
    candidate_collector=_parallel_keyed_table_axis_candidates,
)


class ParallelKeyedTableAndFamilyDetector(
    CrossModuleCollectorCandidateDetector[ParallelKeyedTableAndFamilyCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Keyed table and keyed family should collapse into one auto-registered axis family",
        "The docs require one authoritative owner per closed semantic axis. When a module keeps one keyed table of per-case records and a second keyed nominal family over the same cases, the axis is split across data and behavior. If the family already carries the runtime behavior boundary, the table should derive from that family instead of competing with it.",
        "single authoritative metaclass-registry axis family with derived table/view projections",
        "same enum/key axis is encoded by both a keyed table and a keyed nominal family",
        _AUTHORITATIVE_AUTHORITATIVE_DISPATCH_NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS,
        _CLASS_FAMILY_BUILDER_CALL_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, table_candidate: ParallelKeyedTableAndFamilyCandidate
    ) -> RefactorFinding:
        shape_clause = (
            ""
            if table_candidate.value_shape_name is None
            else f" of `{table_candidate.value_shape_name}` records"
        )
        return self.build_finding(
            (
                f"Axis `{table_candidate.key_type_name}` is split between keyed table `{table_candidate.table_name}`"
                f"{shape_clause} and keyed family `{table_candidate.family_name}` across cases "
                f"{', '.join(table_candidate.shared_case_names[:4])}."
            ),
            table_candidate.evidence,
            scaffold=(
                _axis_policy_registry_scaffold("build(self)")
                + f"\n\n@dataclass(frozen=True)\nclass DerivedAxisRow:\n    key: {_AXIS_POLICY_KEY_TYPE_NAME}\n    policy_type: type[{_AXIS_POLICY_ROOT_NAME}]\n    config: object\n\ndef build_axis_rows() -> tuple[DerivedAxisRow, ...]:\n    return tuple(\n        DerivedAxisRow(key=key, policy_type=policy_type, config=...)\n        for key, policy_type in {_AXIS_POLICY_ROOT_NAME}.__registry__.items()\n    )"
            ),
            codemod_patch=(
                f"# Collapse `{table_candidate.table_name}` and `{table_candidate.family_name}` onto one authoritative metaclass-registry family.\n"
                "# Keep the runtime boundary on the auto-registered family and derive any keyed rows/views from `AxisPolicy.__registry__`."
            ),
            metrics=DISPATCH_ALGEBRA_AUTHORITY.axis_dispatch_metrics(
                table_candidate.shared_case_names,
                table_candidate.key_type_name,
            ),
        )


class CallableMethodAxisRegistryDetector(IssueDetector):
    detector_id = "callable_method_axis_registry"
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_STRATEGY_FAMILY,
        "Callable method-axis registry should become an auto-registered strategy family",
        "A builder call that maps method-axis member names to callable behavior is a hardcoded strategy family in registry-table form. The canonical shape is an ABC plus `AutoRegisterMeta`, with each method implementation declared as a subclass and dispatch routed through `Family.__registry__[method].run(...)`.",
        "AutoRegisterMeta-backed strategy family instead of callable method-axis registry",
        "module-level registry builder maps closed method-axis cases to callable behavior",
        _AUTHORITATIVE_DISPATCH_NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS,
        _BUILDER_CALL_DATAFLOW_ROOT_CLASS_FAMILY_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for module in modules:
            for statement in module.module.body:
                assignment = self._assignment_target_name(statement)
                value = self._assignment_value(statement)
                if assignment is None or not isinstance(value, ast.Call):
                    continue
                if not self._is_method_axis_registry_call(value):
                    continue
                axis_name = ast.unparse(value.args[0]) if value.args else "Axis"
                operation_names = tuple(
                    keyword.arg
                    for keyword in value.keywords
                    if keyword.arg is not None
                    and isinstance(keyword.value, (ast.Name, ast.Attribute))
                )
                if len(operation_names) < 2:
                    continue
                operations = ", ".join(operation_names[:4])
                findings.append(
                    self.build_finding(
                        (
                            f"`{assignment}` maps `{axis_name}` member names to callable operations "
                            f"{operations}; this is a hardcoded strategy family."
                        ),
                        (
                            SourceLocation(
                                str(module.path), statement.lineno, assignment
                            ),
                        ),
                        scaffold=(
                            "from abc import ABC, abstractmethod\n"
                            "from typing import ClassVar\n"
                            "from metaclass_registry import AutoRegisterMeta\n\n"
                            "class MethodStrategy(ABC, metaclass=AutoRegisterMeta):\n"
                            '    __registry_key__ = "method"\n'
                            "    __skip_if_no_key__ = True\n"
                            "    method: ClassVar[object | None] = None\n\n"
                            "    @abstractmethod\n"
                            "    def run(self, request): ...\n\n"
                            "def run_method(method, request):\n"
                            "    return MethodStrategy.__registry__[method].run(request)\n"
                        ),
                        codemod_patch=(
                            f"# Replace callable registry `{assignment}` with an AutoRegisterMeta-backed strategy family.\n"
                            "# Move each callable into a registered subclass and dispatch with `Family.__registry__[method].run(...)`."
                        ),
                        metrics=DispatchCountMetrics.from_literal_family(
                            dispatch_axis=axis_name,
                            literal_cases=operation_names,
                        ),
                    )
                )
        return findings

    @staticmethod
    def _is_method_axis_registry_call(call: ast.Call) -> bool:
        if len(call.args) != 1 or len(call.keywords) < 2:
            return False
        func_name = ast.unparse(call.func)
        if not (
            func_name.endswith(".from_member_names")
            or func_name.endswith("from_member_names")
        ):
            return False
        axis_name = ast.unparse(call.args[0])
        return axis_name.endswith("Method") or axis_name.endswith("Axis")

    @staticmethod
    def _assignment_target_name(statement: ast.stmt) -> str | None:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                return target.id
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            return statement.target.id
        return None

    @staticmethod
    def _assignment_value(statement: ast.stmt) -> ast.AST | None:
        if isinstance(statement, ast.Assign):
            return statement.value
        if isinstance(statement, ast.AnnAssign):
            return statement.value
        return None


class InheritedAutoRegisterConfigBoilerplateDetector(IssueDetector):
    detector_id = "inherited_autoregister_config_boilerplate"
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "AutoRegister root repeats inherited registry configuration",
        "An AutoRegisterMeta root that directly repeats registry protocol fields already declared by a base is carrying boilerplate instead of relying on inheritance. The registry key, skip policy, and related protocol fields should be inherited from the shared nominal base. If AutoRegisterMeta cannot honor inherited registry config, fix the metaclass package rather than repeating the fields on every root.",
        "inherited AutoRegister registry protocol configuration",
        "AutoRegisterMeta class repeats registry protocol assignments from an inherited base",
        _CLASS_LEVEL_REGISTRATION_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
        _CLASS_FAMILY_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        class_index = build_class_family_index(modules)
        findings: list[RefactorFinding] = []
        for indexed_class in sorted(
            class_index.classes_by_symbol.values(), key=lambda item: item.symbol
        ):
            node = indexed_class.node
            if not HELPER_SUPPORT_PROJECTION_AUTHORITY.declares_autoregister_meta(
                node
            ):
                continue
            repeated_fields = self._repeated_inherited_fields(
                class_index, indexed_class
            )
            if not repeated_fields:
                continue
            field_list = ", ".join(repeated_fields)
            findings.append(
                self.build_finding(
                    (
                        f"`{indexed_class.simple_name}` repeats inherited AutoRegister "
                        f"registry field(s) {field_list}."
                    ),
                    (
                        SourceLocation(
                            indexed_class.file_path,
                            indexed_class.line,
                            indexed_class.simple_name,
                        ),
                    ),
                    scaffold=(
                        "class RegisteredFamilyBase(ABC):\n"
                        '    __registry_key__ = "method"\n'
                        "    __skip_if_no_key__ = True\n\n"
                        "class ConcreteFamilyRoot(RegisteredFamilyBase, metaclass=AutoRegisterMeta):\n"
                        "    # declare behavior contract only; inherit registry config\n"
                        "    ..."
                    ),
                    codemod_patch=(
                        f"# Delete repeated registry protocol fields {field_list} from `{indexed_class.simple_name}`.\n"
                        "# Keep those fields on the inherited shared base. If the runtime registry does not honor inherited config, fix AutoRegisterMeta inheritance semantics instead of copying boilerplate."
                    ),
                    metrics=MappingMetrics(
                        mapping_site_count=len(repeated_fields),
                        field_count=len(repeated_fields),
                        mapping_name=indexed_class.simple_name,
                        field_names=repeated_fields,
                    ),
                )
            )
        return findings

    @staticmethod
    def _repeated_inherited_fields(
        class_index: ClassFamilyIndex,
        indexed_class: IndexedClass,
    ) -> tuple[str, ...]:
        protocol_fields = (
            "__key_extractor__",
            "__registry_key__",
            "__skip_if_no_key__",
        )
        direct_assignments = CLASS_NODE_AUTHORITY.direct_assignments(
            indexed_class.node
        )
        repeated: list[str] = []
        for field_name in protocol_fields:
            current_value = direct_assignments.get(field_name)
            if current_value is None:
                continue
            current_text = ast.unparse(current_value)
            for ancestor_symbol in class_index.ancestor_symbols(indexed_class.symbol):
                ancestor = class_index.class_for(ancestor_symbol)
                if ancestor is None:
                    continue
                ancestor_value = CLASS_NODE_AUTHORITY.direct_assignments(
                    ancestor.node
                ).get(field_name)
                if ancestor_value is None:
                    continue
                if ast.unparse(ancestor_value) == current_text:
                    repeated.append(field_name)
                    break
        return tuple(repeated)


class AutoRegisterExplicitPriorityOrderingDetector(IssueDetector):
    detector_id = "autoregister_explicit_priority_ordering"
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "AutoRegister family uses explicit priority ordering instead of MRO",
        "An AutoRegisterMeta family whose registered leaves carry a `priority` class attribute is maintaining a second ordering authority beside the inheritance graph. If ordering is semantic, the nominal hierarchy and MRO should carry it; if ordering is only presentation, it should be a derived view outside the registered family.",
        "MRO-owned ordering for registered semantic families",
        "AutoRegisterMeta family declares or consumes a class-level priority axis to sort registered implementations",
        _CLASS_LEVEL_REGISTRATION_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
        _CLASS_FAMILY_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _collect_findings(
        self, modules: list[ParsedModule], config: DetectorConfig
    ) -> list[RefactorFinding]:
        del config
        class_index = build_class_family_index(modules)
        findings: list[RefactorFinding] = []
        for indexed_class in sorted(
            class_index.classes_by_symbol.values(), key=lambda item: item.symbol
        ):
            node = indexed_class.node
            if not HELPER_SUPPORT_PROJECTION_AUTHORITY.declares_autoregister_meta(
                node
            ):
                continue
            priority_sites = self._priority_sites(class_index, indexed_class)
            if not priority_sites:
                continue
            if not self._sorts_registry_by_priority(
                indexed_class.simple_name,
                node,
                modules,
            ):
                continue
            findings.append(
                self.build_finding(
                    (
                        f"`{indexed_class.simple_name}` orders registered implementations "
                        "through explicit class-level `priority` values."
                    ),
                    (
                        SourceLocation(
                            indexed_class.file_path,
                            indexed_class.line,
                            indexed_class.simple_name,
                        ),
                        *priority_sites[:5],
                    ),
                    scaffold=(
                        "class RegisteredPolicy(ABC, metaclass=AutoRegisterMeta):\n"
                        "    @classmethod\n"
                        "    def ordered(cls):\n"
                        "        return tuple(cls.__subclasses__())\n\n"
                        "# Encode ordering by inheritance/MRO, not by a parallel priority field."
                    ),
                    codemod_patch=(
                        f"# Delete the `priority` class axis from `{indexed_class.simple_name}` and its leaves.\n"
                        "# Replace `sorted(..., key=lambda item: item.priority)` registry traversal with an MRO/subclass traversal owned by the nominal hierarchy."
                    ),
                    metrics=MappingMetrics(
                        mapping_site_count=len(priority_sites),
                        field_count=len(priority_sites),
                        mapping_name=indexed_class.simple_name,
                        field_names=("priority",),
                    ),
                )
            )
        return findings

    def _priority_sites(
        self,
        class_index: ClassFamilyIndex,
        indexed_class: IndexedClass,
    ) -> tuple[SourceLocation, ...]:
        symbols = (
            indexed_class.symbol,
            *class_index.descendant_symbols(indexed_class.symbol),
        )
        sites: list[SourceLocation] = []
        for symbol in symbols:
            candidate = class_index.class_for(symbol)
            if candidate is None:
                continue
            line = _class_level_assignment_line(candidate.node, "priority")
            if line is None:
                continue
            sites.append(SourceLocation(candidate.file_path, line, candidate.simple_name))
        return tuple(sites)

    @staticmethod
    def _sorts_registry_by_priority(
        root_name: str,
        node: ast.ClassDef,
        modules: list[ParsedModule],
    ) -> bool:
        return any(
            _call_sorts_registry_by_priority(child, root_name)
            for child in _walk_nodes(node)
            if isinstance(child, ast.Call)
        ) or any(
            _call_sorts_registry_by_priority(child, root_name)
            for module in modules
            for child in _walk_nodes(module.module)
            if isinstance(child, ast.Call)
        )


def _class_level_assignment_line(node: ast.ClassDef, name: str) -> int | None:
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and name_id(statement.target) == name:
            return statement.lineno
        if isinstance(statement, ast.Assign) and any(
            name_id(target) == name for target in statement.targets
        ):
            return statement.lineno
    return None


def _call_sorts_registry_by_priority(call: ast.Call, root_name: str) -> bool:
    if name_id(call.func) != "sorted":
        return False
    if not any(_node_mentions_registry(node, root_name) for node in call.args):
        return False
    return any(
        keyword.arg == "key"
        and keyword.value is not None
        and _node_mentions_priority_attribute(keyword.value)
        for keyword in call.keywords
    )


def _node_mentions_registry(node: ast.AST, root_name: str) -> bool:
    for child in _walk_nodes(node):
        if not isinstance(child, ast.Attribute) or child.attr != "__registry__":
            continue
        if isinstance(child.value, ast.Name) and child.value.id in {"cls", root_name}:
            return True
    return False


def _node_mentions_priority_attribute(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Attribute) and child.attr == "priority"
        for child in _walk_nodes(node)
    )


class EnumKeyedTableClassAxisShadowDetector(
    ModuleCollectorCandidateDetector[EnumKeyedTableClassAxisShadowCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Enum-keyed table should derive from auto-registered class-declared axis keys",
        "The docs require a single writable owner per closed semantic axis. If a module already declares that axis through class-level enum assignments, adding a writable enum-keyed table over the same cases creates duplicate authority and a synchronization surface. The class-declared axis should be the primary owner and any enum-keyed lookup should be derived from the family registry.",
        "one authoritative metaclass-registry closed-axis owner with derived table/view projections",
        "module-level enum-keyed table overlaps a class family that already declares the same enum axis",
        _AUTHORITATIVE_CLOSED_FAMILY_DISPATCH_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _PROJECTION_DICT_CLASS_FAMILY_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, axis_candidate: EnumKeyedTableClassAxisShadowCandidate
    ) -> RefactorFinding:
        class_names = ", ".join(axis_candidate.class_names[:4])
        shared_cases = ", ".join(axis_candidate.shared_case_names[:4])
        value_names = ", ".join(axis_candidate.value_type_names[:4])
        return self.build_finding(
            (
                f"`{axis_candidate.table_name}` maps `{axis_candidate.key_type_name}` cases {shared_cases} "
                f"to {value_names}, while classes {class_names} already declare the same axis via "
                f"`{axis_candidate.key_attr_name}`."
            ),
            axis_candidate.evidence,
            scaffold=(
                _axis_policy_registry_scaffold("route_type(self)")
                + f"\n\nAXIS_BY_KEY = {{\n    key: policy_type\n    for key, policy_type in {_AXIS_POLICY_ROOT_NAME}.__registry__.items()\n}}\n"
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


@dataclass(frozen=True)
class EnumConstructorPolicyTable:
    """One enum-keyed dict that constructs behavioral policy objects."""

    file_path: str
    line: int
    owner_symbol: str
    table_name: str
    enum_name: str
    case_names: tuple[str, ...]
    constructor_names: tuple[str, ...]

    @property
    def evidence(self) -> tuple[SourceLocation, ...]:
        return (
            SourceLocation(
                self.file_path,
                self.line,
                f"{self.owner_symbol}.{self.table_name}",
            ),
        )


def _enum_member_key(node: ast.AST) -> tuple[str, str] | None:
    if not isinstance(node, ast.Attribute):
        return None
    if not isinstance(node.value, ast.Name):
        return None
    return (node.value.id, node.attr)


def _constructor_call_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _assignment_target_name(node: ast.Assign | ast.AnnAssign) -> str | None:
    if isinstance(node, ast.AnnAssign):
        return name_id(node.target)
    for target in node.targets:
        name = name_id(target)
        if name is not None:
            return name
    return None


def _looks_like_behavior_policy_constructor(name: str) -> bool:
    return name.endswith(("Policy", "Strategy", "Handler", "Route", "Runner"))


def _enum_constructor_policy_table_from_assignment(
    *,
    module: ParsedModule,
    node: ast.Assign | ast.AnnAssign,
    owner_symbol: str,
) -> EnumConstructorPolicyTable | None:
    value = as_ast(node.value, ast.Dict)
    if value is None or len(value.keys) < 2:
        return None

    key_pairs = tuple(
        pair for key in value.keys if (pair := _enum_member_key(key)) is not None
    )
    if len(key_pairs) != len(value.keys):
        return None
    enum_names = {enum_name for enum_name, _case_name in key_pairs}
    if len(enum_names) != 1:
        return None

    constructor_names = tuple(
        name
        for value_node in value.values
        if (name := _constructor_call_name(value_node)) is not None
    )
    if len(constructor_names) != len(value.values):
        return None
    if len(set(constructor_names)) < 2:
        return None
    if not any(
        _looks_like_behavior_policy_constructor(name) for name in constructor_names
    ):
        return None

    table_name = _assignment_target_name(node) or "enum_constructor_policy_table"
    enum_name = next(iter(enum_names))
    return EnumConstructorPolicyTable(
        file_path=str(module.path),
        line=node.lineno,
        owner_symbol=owner_symbol,
        table_name=table_name,
        enum_name=enum_name,
        case_names=tuple(case_name for _enum_name, case_name in key_pairs),
        constructor_names=constructor_names,
    )


def _enum_constructor_policy_tables(
    module: ParsedModule,
) -> tuple[EnumConstructorPolicyTable, ...]:
    tables: list[EnumConstructorPolicyTable] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.owner_stack: list[str] = ["<module>"]

        @property
        def owner_symbol(self) -> str:
            return ".".join(self.owner_stack)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.owner_stack.append(node.name)
            self.generic_visit(node)
            self.owner_stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.owner_stack.append(node.name)
            self.generic_visit(node)
            self.owner_stack.pop()

        def visit_Assign(self, node: ast.Assign) -> None:
            self._visit_assignment(node)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            self._visit_assignment(node)
            self.generic_visit(node)

        def _visit_assignment(self, node: ast.Assign | ast.AnnAssign) -> None:
            table = _enum_constructor_policy_table_from_assignment(
                module=module,
                node=node,
                owner_symbol=self.owner_symbol,
            )
            if table is not None:
                tables.append(table)

    Visitor().visit(module.module)
    return tuple(tables)


class ManualEnumConstructorPolicyTableDetector(PerModuleIssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Enum-keyed constructor policy table should be auto-registered",
        "A dict that maps enum cases directly to concrete policy/strategy/handler constructor calls is a manually synchronized closed behavioral axis. The enum case and concrete implementation membership should be declared by the implementation class and collected by an AutoRegisterMeta-backed nominal family.",
        "AutoRegisterMeta-backed policy family with class-declared enum keys",
        "enum-keyed dict literal constructs concrete behavioral policy instances",
        _AUTHORITATIVE_CLOSED_FAMILY_DISPATCH_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _PROJECTION_DICT_CLASS_FAMILY_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _findings_for_module(
        self,
        module: ParsedModule,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        findings: list[RefactorFinding] = []
        for table in _enum_constructor_policy_tables(module):
            constructor_summary = ", ".join(table.constructor_names[:4])
            case_summary = ", ".join(table.case_names[:4])
            findings.append(
                self.build_finding(
                    (
                        f"`{table.owner_symbol}` builds `{table.table_name}` by mapping "
                        f"`{table.enum_name}` cases {case_summary} to constructor calls "
                        f"{constructor_summary}."
                    ),
                    table.evidence,
                    scaffold=(
                        "from abc import ABC, abstractmethod\n"
                        "from typing import ClassVar\n"
                        "from metaclass_registry import AutoRegisterMeta\n\n"
                        "class RegisteredPolicy(ABC, metaclass=AutoRegisterMeta):\n"
                        "    __registry_key__ = 'policy_key'\n"
                        "    __skip_if_no_key__ = True\n"
                        "    policy_key: ClassVar[EnumCase | None] = None\n\n"
                        "    @abstractmethod\n"
                        "    def run(self, request): ..."
                    ),
                    codemod_patch=(
                        f"# Delete manual enum-keyed policy table `{table.table_name}`.\n"
                        f"# Give each policy class a `{table.enum_name}` key and dispatch through the AutoRegisterMeta registry."
                    ),
                    metrics=MappingMetrics(
                        mapping_site_count=len(table.case_names),
                        field_count=1,
                        mapping_name=table.table_name,
                        field_names=table.constructor_names,
                        source_name=table.enum_name,
                        identity_field_names=("policy_key",),
                    ),
                )
            )
        return findings


class TransportShellTemplateMethodDetector(
    ConfiguredModuleCollectorCandidateDetector[TransportShellTemplateCandidate]
):
    candidate_collector = _transport_shell_template_candidates
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Template-method family is a transport shell over a downstream authority",
        "The docs say nominal families should have one authoritative owner. When an ABC template method only materializes an intermediate object from a class-level selector, delegates through one hook, and repackages through another hook, the extra family is usually a transport shell around an already authoritative boundary.",
        "single authoritative materialization/execution family instead of a parallel transport shell",
        "template family varies mostly by class-level selector and result adapter",
        _AUTHORITATIVE_SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _CLASS_FAMILY_BUILDER_CALL_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, shell_candidate: TransportShellTemplateCandidate
    ) -> RefactorFinding:
        selector_values = ", ".join(shell_candidate.selector_value_names)
        kwargs_clause = (
            f" plus `{shell_candidate.kwargs_helper_name}({shell_candidate.source_param_name})`"
            if shell_candidate.kwargs_helper_name is not None
            else ""
        )
        return self.build_finding(
            (
                f"`{shell_candidate.class_name}.{shell_candidate.driver_method_name}` materializes selector values "
                f"{selector_values} from `{shell_candidate.selector_attr_name}` via `{shell_candidate.constructor_name}`"
                f"{kwargs_clause} across {len(shell_candidate.concrete_class_names)} concrete leaves, then only delegates "
                f"through `{shell_candidate.inner_hook_name}` and `{shell_candidate.outer_hook_name}`."
            ),
            (shell_candidate.evidence,),
            scaffold=(
                "@dataclass(frozen=True)\nclass MaterializationSpec:\n    selector: object\n    materializer: object\n    executor: object\n    packager: object\n# Dispatch once on the authoritative selector/spec family."
            ),
            codemod_patch=(
                f"# Collapse `{shell_candidate.class_name}` onto the downstream selector/spec family.\n"
                "# Keep one selection boundary and let that boundary own materialization, execution, and result packaging."
            ),
        )


class CrossModuleSpecAxisAuthorityDetector(
    ConfiguredCrossModuleCollectorCandidateDetector[
        CrossModuleSpecAxisAuthorityCandidate
    ]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Cross-module spec axis should have one authority",
        "The docs say one semantic family should have one authoritative owner. When two modules encode the same identity-axis -> executable-axis spec pairs, one table is a duplicate authority unless it is explicitly derived.",
        "one repository-wide authoritative spec-axis family",
        "same identity/executable spec axis is re-encoded across modules",
        _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _BUILDER_CALL_DATAFLOW_ROOT_CLASS_FAMILY_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, authority_candidate: CrossModuleSpecAxisAuthorityCandidate
    ) -> RefactorFinding:
        family_names = ", ".join(
            (
                f"{Path(family.file_path).name}:{family.family_name}"
                for family in authority_candidate.families
            )
        )
        pair_names = ", ".join(
            (
                f"{identity}->{executable}"
                for identity, executable in authority_candidate.shared_axis_pairs
            )
        )
        axis_fields = " -> ".join(authority_candidate.axis_field_names)
        evidence = tuple(
            (family.evidence for family in authority_candidate.families[:6])
        )
        return self.build_finding(
            (
                f"Families {family_names} each encode the same `{axis_fields}` pairs {pair_names} across module boundaries."
            ),
            evidence,
            scaffold=(
                "@dataclass(frozen=True)\nclass AxisExecutionSpec:\n    identity: object\n    executable: object\n# Keep one exported authority and let downstream modules compose from it."
            ),
            codemod_patch=(
                "# Extract one repository-wide spec-axis family.\n# Make downstream wrappers, benchmarks, or adapters reference that authority instead of restating identity/executable pairs."
            ),
        )


class ParallelRegistryProjectionFamilyDetector(
    ModuleCollectorCandidateDetector[ParallelRegistryProjectionFamilyCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Parallel registry projection builders should collapse into one family spec",
        "The docs say one semantic family should have one authoritative owner. When several functions differ only in which registry authority feeds which target constructor, the projection-axis mapping should become one declared spec or family authority instead of several hand-wired wrappers.",
        "single authoritative registry-projection family",
        "same registry-authority-to-target projection shape repeated across sibling functions",
        _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _BUILDER_CALL_CLASS_FAMILY_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, catalog_candidate: ParallelRegistryProjectionFamilyCandidate
    ) -> RefactorFinding:
        function_names = ", ".join(
            (function.qualname for function in catalog_candidate.functions[:4])
        )
        extractor_bases = ", ".join(
            (
                function.extractor_base_name
                for function in catalog_candidate.functions[:4]
            )
        )
        catalog_types = ", ".join(
            (function.catalog_type_name for function in catalog_candidate.functions[:4])
        )
        evidence = tuple(
            function.evidence for function in catalog_candidate.functions[:6]
        )
        return self.build_finding(
            (
                f"Functions {function_names} each build {catalog_types} through "
                f"`{catalog_candidate.collector_name}(structure, ExtractorBase.{catalog_candidate.registry_accessor_name}())` "
                f"over parallel extractor bases {extractor_bases}."
            ),
            evidence,
            scaffold=(
                "@dataclass(frozen=True)\nclass RegistryProjectionSpec:\n    registry_authority: type\n    target_type: type\n# One helper should own the registry-authority to target mapping."
            ),
            codemod_patch=(
                "# Extract one registry-projection family spec and one authoritative projection builder.\n# Make per-axis public helpers delegate to that authority instead of reconstructing collector(...registry_accessor())."
            ),
        )


class RepeatedKeyedFamilyDetector(
    ConfiguredCrossModuleCollectorCandidateDetector[RepeatedKeyedFamilyCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Repeated keyed family scaffolding should collapse into one typed metaclass-registry base",
        "The docs encourage aggressive metaprogramming when several nominal families repeat the same class-level registration and lookup shell. When many roots restate `registry_key_attr`, `_registry`, and `for_*` lookup methods, the family algorithm should live in one typed `metaclass-registry` base.",
        "single typed metaclass-registry substrate for keyed nominal registries",
        "same keyed family registration and lookup shell repeated across nominal family roots",
        _CLASS_LEVEL_REGISTRATION_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
        _CLASS_FAMILY_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, family_candidate: RepeatedKeyedFamilyCandidate
    ) -> RefactorFinding:
        class_names = ", ".join(
            (root.class_name for root in family_candidate.roots[:8])
        )
        lookup_names = ", ".join(
            sorted({root.lookup_method_name for root in family_candidate.roots[:8]})
        )
        registry_keys = ", ".join(
            sorted({root.registry_key_attr_name for root in family_candidate.roots[:8]})
        )
        evidence = tuple(root.evidence for root in family_candidate.roots[:8])
        return self.build_finding(
            (
                f"Registry roots {class_names} each repeat `{registry_keys}` + `_registry` + "
                f"`{lookup_names}` over `{family_candidate.family_base_name}`."
            ),
            evidence,
            scaffold=(
                'from metaclass_registry import AutoRegisterMeta\n\nKeyT = TypeVar("KeyT")\n\nclass KeyedNominalFamily(ABC, Generic[KeyT], metaclass=AutoRegisterMeta):\n    __registry_key__ = "registry_key"\n    __skip_if_no_key__ = True\n    registry_key: ClassVar[KeyT | None] = None\n    family_label: ClassVar[str] = "family"\n    @classmethod\n    def for_key(cls, key: KeyT):\n        try:\n            return cls.__registry__[key]\n        except KeyError as error:\n            raise ValueError(f"Unknown {cls.family_label}: {key}") from error'
            ),
            codemod_patch=(
                "# Extract one typed metaclass-registry base that owns registration lookup, duplicate handling, and error shaping.\n# Leave only declarative key attributes and irreducible hook methods on each family root, and read the registered classes from `cls.__registry__`."
            ),
        )


def _registry_maturity_fanout_metrics(
    candidate: PrematureRegistryInfrastructureCandidate,
) -> RegistrationMetrics:
    return RegistrationMetrics(
        registration_site_count=len(candidate.registered_case_names),
        registry_name=candidate.class_name,
    )


declare_candidate_rule_detector(
    NonInjectiveTypeRegistryCandidate,
    high_confidence_certified_spec(
        PatternId.AUTO_REGISTER_META,
        "Type registry must be injective over its key axis",
        "A nominal registry is only type-safe when each concrete implementation has one canonical key and each key resolves to one implementation. Duplicate keys, duplicate type identities, or concrete descendants without keys mean the registry cannot serve as an injective authority.",
        "injective type registry with one stable key per concrete implementation",
        "registry key axis aliases multiple implementation types or misses concrete descendants",
        _CLASS_LEVEL_REGISTRATION_AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,
        _CLASS_FAMILY_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}` registry axis `{candidate.key_type_name}` is not injective: "
        f"duplicate keys {candidate.duplicate_key_names}, duplicate types "
        f"{candidate.duplicate_type_names}, missing keyed types {candidate.missing_type_names}."
    ),
    evidence=lambda candidate: candidate.evidence,
    scaffold=lambda candidate: (
        "@dataclass(frozen=True)\nclass InjectiveRegistryRow:\n    key: object\n    implementation_type: type[object]\n\n"
        "# Build the registry from rows only after proving keys and implementation types are one-to-one."
    ),
    codemod_patch=lambda candidate: (
        f"# Repair `{candidate.class_name}` before adding or keeping registry metaprogramming.\n"
        "# Give every concrete implementation exactly one canonical key and delete aliases or duplicate key writes.\n"
        "# If aliases are semantic, model them as an explicit alias projection instead of a second registry identity."
    ),
    metrics=lambda candidate: RegistrationMetrics(
        registration_site_count=len(candidate.registered_case_names),
        registry_name=candidate.class_name,
    ),
    detector_base=ConfiguredCrossModuleCollectorCandidateDetector,
    candidate_collector=_non_injective_type_registry_candidates,
)


declare_candidate_rule_detector(
    InjectiveTypeRegistryCandidate,
    high_confidence_certified_spec(
        PatternId.AUTO_REGISTER_META,
        "Mature injective type registry should use metaclass registration",
        "A registry with a stable key axis, lookup lifecycle, consumer fanout, and an injective type-to-key proof has reached the point where handwritten registration mechanics are declaration noise. The metaclass should own population while implementation classes declare only their canonical key and behavior hooks.",
        "AutoRegisterMeta-backed ABC with an injective type-key proof",
        "registry axis proves one key per implementation type plus mature lookup and consumer fanout",
        _CLASS_LEVEL_REGISTRATION_AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,
        _CLASS_FAMILY_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}` is a mature injective registry over `{candidate.key_type_name}`: "
        f"keys {candidate.registered_case_names}, lookup {candidate.lookup_method_names}, "
        f"consumers {candidate.consumer_symbols}; replace handwritten registry mechanics with AutoRegisterMeta."
    ),
    evidence=lambda candidate: candidate.evidence,
    scaffold=lambda candidate: _metaclass_registry_keyed_family_scaffold(
        root_name="InjectiveRegistryFamily",
        key_attr_name=candidate.registry_key_attr_name,
        key_type_name=candidate.key_type_name,
        method_defs=("run(self)",),
    ),
    codemod_patch=lambda candidate: (
        f"# Replace `{candidate.class_name}` handwritten `_registry` population with `AutoRegisterMeta`.\n"
        f"# Keep `{candidate.registry_key_attr_name}` as the canonical class-level key and let the metaclass prove class-time population."
    ),
    metrics=lambda candidate: RegistrationMetrics(
        registration_site_count=len(candidate.registered_case_names),
        registry_name=candidate.class_name,
    ),
    detector_base=ConfiguredCrossModuleCollectorCandidateDetector,
    candidate_collector=_injective_type_registry_candidates,
)


declare_candidate_rule_detector(
    RegistryProjectionSurfaceCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Manual registry projection surfaces should derive from the injective registry",
        "Once a registry proves one canonical key per implementation type, export rosters, key/type maps, and option lists are projections of that registry authority. Hand-maintaining those surfaces creates shadow authorities that can drift away from the type-safe registry.",
        "generated projection surface derived from an injective registry proof",
        "manual list or dict surface repeats keys/types already proven by an injective registry",
        _AUTHORITATIVE_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
        _CLASS_FAMILY_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: (
        f"`{candidate.surface_name}` is a manual `{candidate.projection_role}` "
        f"`{candidate.surface_kind}` projection "
        f"of injective registry `{candidate.registry_class_name}` over `{candidate.key_type_name}`: "
        f"keys {candidate.shared_key_names}, types {candidate.shared_type_names}, "
        f"coverage {candidate.projection_coverage_ratio:.2f}; "
        f"target `{candidate.projection_target_name}`, "
        f"materialization `{candidate.materialization_rule}`, "
        f"decompression key `{candidate.decompression_key}`."
        + (
            f" Subset policy hint `{candidate.subset_policy_hint}` names the quotient; repeated use should be owned by a projection policy authority."
            if candidate.subset_policy_hint is not None
            and candidate.projection_coverage_ratio < 1.0
            else (
                f" Missing keys {candidate.missing_key_names} and types {candidate.missing_type_names} need a named projection policy."
                if candidate.projection_coverage_ratio < 1.0
                else ""
            )
        )
    ),
    evidence=lambda candidate: (
        SourceLocation(candidate.file_path, candidate.line, candidate.surface_name),
    ),
    scaffold=lambda candidate: (
        "@dataclass(frozen=True)\n"
        "class RegistryProjectionSpec:\n"
        "    registry_authority: type[object]\n"
        "    projection_policy: str\n"
        "    projection_target: str\n"
        "    materialization_rule: str\n"
        "    decompression_key: str\n\n"
        "def derive_registry_projection(spec: RegistryProjectionSpec):\n"
        "    return project_from_injective_registry(\n"
        "        spec.registry_authority,\n"
        "        policy=spec.projection_policy,\n"
        "        target=spec.projection_target,\n"
        "        materialization=spec.materialization_rule,\n"
        "    )"
    ),
    codemod_patch=lambda candidate: (
        f"# Delete `{candidate.surface_name}` as a handwritten `{candidate.projection_role}` `{candidate.surface_kind}`.\n"
        f"# Replace it with RegistryProjectionSpec({candidate.registry_class_name}, policy={candidate.projection_policy_name!r}, target={candidate.projection_target_name!r}, materialization={candidate.materialization_rule!r}).\n"
        + (
            f"# Its decompression key is `{candidate.decompression_key}`; derive it from the injective key/type registry proof."
            if candidate.projection_coverage_ratio >= 1.0
            else (
                f"# Its decompression key is `{candidate.decompression_key}`; derive it through an explicit `{candidate.subset_policy_hint}` projection policy."
                if candidate.subset_policy_hint is not None
                else f"# Either derive the full surface from `{candidate.registry_class_name}` or add a named projection policy explaining the missing keys/types."
            )
        )
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(candidate.projected_names),
        mapping_name=candidate.surface_name,
        field_names=(
            candidate.registry_class_name,
            candidate.key_type_name,
            candidate.projection_policy_name,
            candidate.projection_target_name,
            candidate.materialization_rule,
        ),
    ),
    detector_base=ConfiguredCrossModuleCollectorCandidateDetector,
    candidate_collector=_REGISTRY_PROJECTION_SURFACE_ANALYZER.surface_candidates,
)


declare_candidate_rule_detector(
    RegistryProjectionPolicyAuthorityCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated registry subset projections should share a nominal policy authority",
        "A partial projection of an injective registry is a quotient of the registry axis. When several surfaces repeat the same quotient hint, the hint should become a first-class projection policy instead of living as independent allowlists.",
        "nominal registry projection policy reused by generated subset surfaces",
        "multiple registry projection surfaces repeat the same subset hint without one owner",
        _AUTHORITATIVE_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
        _CLASS_FAMILY_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: (
        f"`{candidate.registry_class_name}` has repeated `{candidate.policy_hint}` subset projections "
        f"{candidate.surface_names} across roles {candidate.surface_roles}; move the quotient into one policy authority "
        f"and materialize targets {candidate.projection_target_names} from specs."
    ),
    evidence=lambda candidate: candidate.evidence,
    scaffold=lambda candidate: (
        "class RegistryProjectionPolicy(ABC):\n"
        "    @abstractmethod\n"
        "    def includes_key(self, key): ...\n\n"
        f"class {candidate.policy_hint.title()}ProjectionPolicy(RegistryProjectionPolicy):\n"
        "    def includes_key(self, key): ...\n\n"
        "REGISTRY_PROJECTION_SPECS = (...,)"
    ),
    codemod_patch=lambda candidate: (
        f"# Replace repeated `{candidate.policy_hint}` subset surfaces {candidate.surface_names} with one nominal projection policy.\n"
        f"# Generate specs for targets {candidate.projection_target_names} using decompression keys {candidate.decompression_keys}."
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(candidate.surface_names),
        mapping_name=f"{candidate.policy_hint}_projection_policy",
        field_names=(
            candidate.registry_class_name,
            candidate.key_type_name,
            *candidate.surface_roles,
            *candidate.materialization_rules,
        ),
    ),
    detector_base=ConfiguredCrossModuleCollectorCandidateDetector,
    candidate_collector=(
        _REGISTRY_PROJECTION_SURFACE_ANALYZER.policy_authority_candidates
    ),
)


declare_candidate_rule_detector(
    PrematureRegistryInfrastructureCandidate,
    high_confidence_certified_spec(
        PatternId.AUTO_REGISTER_META,
        "Registry infrastructure should prove key, lifecycle, and fanout maturity",
        "The OpenHCS history showed that registries pay rent only when the key axis is stable, registration lifecycle is explicit, and more than one consumer uses the registry. A registry-shaped class without those signals is likely a premature abstraction boundary.",
        "mature registry authority with stable key axis, class-time lifecycle, and consumer fanout",
        "keyed registry infrastructure exists before registered cases and consumers prove the axis",
        _CLASS_LEVEL_REGISTRATION_AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,
        _CLASS_FAMILY_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: (
        f"`{candidate.class_name}` is registry-shaped over `{candidate.key_type_name}` via "
        f"`{candidate.registry_key_attr_name}`, but missing maturity signals "
        f"{candidate.missing_maturity_signals}: cases {candidate.registered_case_names}, "
        f"lookup methods {candidate.lookup_method_names}, consumers {candidate.consumer_symbols}."
    ),
    evidence=lambda candidate: candidate.evidence,
    scaffold=lambda candidate: (
        "@dataclass(frozen=True)\nclass AxisRow:\n    key: object\n    value: object\n\n"
        "# Keep rows in a small typed table until key cases, lifecycle, and consumer fanout are stable enough for a registry."
    ),
    codemod_patch=lambda candidate: (
        f"# Do not promote `{candidate.class_name}` to registry infrastructure until it proves all three signals:\n"
        "# stable key cases, explicit lookup/class-time lifecycle, and at least two independent consumers.\n"
        "# Replace premature registry infrastructure with a typed table or local strategy map while any signal is missing."
    ),
    metrics=_registry_maturity_fanout_metrics,
    detector_base=ConfiguredCrossModuleCollectorCandidateDetector,
    candidate_collector=_premature_registry_infrastructure_candidates,
)


class ManualKeyedRecordTableDetector(
    ConfiguredModuleCollectorCandidateDetector[ManualKeyedRecordTableGroupCandidate]
):
    candidate_collector = _manual_keyed_record_table_group_candidates
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Manual keyed record tables should collapse into one authoritative spec table",
        "When several frozen record classes repeat `_registry`, `register`, and `for_*` lookup around closed keys, the code is hand-maintaining multiple writable tables. The docs prefer one authoritative spec tuple or generic keyed-record table with derived indexes.",
        "single authoritative keyed-record table or derived index",
        "same manual record registration and keyed lookup shell repeated across data classes",
        _AUTHORITATIVE_CLOSED_FAMILY_DISPATCH_PROVENANCE_CAPABILITY_TAGS,
        _BUILDER_CALL_DATAFLOW_ROOT_CLASS_FAMILY_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, group_candidate: ManualKeyedRecordTableGroupCandidate
    ) -> RefactorFinding:
        class_names = ", ".join(
            (item.class_name for item in group_candidate.classes[:6])
        )
        key_fields = ", ".join(
            sorted({item.key_field_name for item in group_candidate.classes[:6]})
        )
        lookup_names = ", ".join(
            sorted({item.lookup_method_name for item in group_candidate.classes[:6]})
        )
        evidence = tuple(item.evidence for item in group_candidate.classes[:6])
        return self.build_finding(
            (
                f"Record tables {class_names} each repeat `_registry`, `{group_candidate.classes[0].register_method_name}`, "
                f"and `{lookup_names}` around key fields {key_fields}."
            ),
            evidence,
            scaffold=(
                'KeyT = TypeVar("KeyT")\nRecordT = TypeVar("RecordT")\n\n@dataclass(frozen=True)\nclass KeyedRecordTable(Generic[KeyT, RecordT]):\n    records: tuple[RecordT, ...]\n    key_of: Callable[[RecordT], KeyT]\n\n    def by_key(self) -> dict[KeyT, RecordT]:\n        return {self.key_of(record): record for record in self.records}'
            ),
            codemod_patch=(
                "# Replace per-class mutable `_registry` + `register` shells with one authoritative tuple of record specs.\n# Derive the keyed lookup dict once, or factor the pattern into a generic keyed-record table helper."
            ),
        )


class ManualStructuralRecordMechanicsDetector(
    ConfiguredModuleCollectorCandidateDetector[
        ManualStructuralRecordMechanicsGroupCandidate
    ]
):
    candidate_collector = _manual_structural_record_mechanics_group_candidates
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated structural record mechanics should derive from field metadata",
        "When several frozen dataclass records hand-write validation, tuple-style field projection, round-trip reconstruction, and fieldwise transform logic, those mechanics have become a second authority beside the field declarations. The docs prefer one metadata-driven record substrate that derives those mechanics from typed fields.",
        "single typed structural-record substrate with derived validation, projection, and transform mechanics",
        "same dataclass record lifecycle mechanics repeated across sibling structural record classes",
        _AUTHORITATIVE_FAIL_LOUD_CONTRACTS_PROVENANCE_TYPE_LINEAGE_CAPABILITY_TAGS,
        _CLASS_FAMILY_DATAFLOW_ROOT_BUILDER_CALL_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, group_candidate: ManualStructuralRecordMechanicsGroupCandidate
    ) -> RefactorFinding:
        class_names = ", ".join(
            (item.class_name for item in group_candidate.classes[:6])
        )
        shared_methods = ", ".join(group_candidate.shared_method_names)
        transform_methods = ", ".join(group_candidate.transform_method_names[:6])
        base_names = ", ".join(group_candidate.base_names)
        evidence = tuple(item.evidence for item in group_candidate.classes[:6])
        return self.build_finding(
            (
                f"Dataclass records {class_names} each hand-roll `{shared_methods}` plus fieldwise transforms "
                f"{transform_methods} on top of base family `{base_names}`."
            ),
            evidence,
            scaffold=(
                "@dataclass_transform(field_specifiers=(field, record_field))\nclass StructuralRecordBase:\n    def validate(self): ...\n    def project_fields(self): ...\n    @classmethod\n    def from_projected(cls, projected, metadata): ...\n    def transformed(self, **changes): ...\n"
            ),
            codemod_patch=(
                "# Move validation constraints, projected-field partitions, and transform semantics into typed field metadata.\n# Derive projection, round-trip reconstruction, and fieldwise transforms from one structural-record base instead of re-encoding them per class."
            ),
        )


class RepeatedConcreteTypeCaseAnalysisDetector(
    ConfiguredCrossModuleCollectorCandidateDetector[
        RepeatedConcreteTypeCaseAnalysisCandidate
    ]
):
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_INTERFACE_WITNESS,
        "Repeated concrete-type recovery should become nominal family behavior",
        "When several functions repeatedly recover the same semantic family through concrete `isinstance` checks on one carried attribute, the family boundary is still latent. The docs want one nominal ABC and concrete leaf behavior exposed through typed properties or hooks instead of repeated leaf decoding.",
        "single ABC-backed family for the carried subject, with repeated case recovery moved into nominal properties or hooks",
        "same attribute-carried family is re-decoded through repeated concrete runtime type checks across several functions",
        _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_MRO_ORDERING_CAPABILITY_TAGS,
        _CLASS_FAMILY_DATAFLOW_ROOT_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, case_candidate: RepeatedConcreteTypeCaseAnalysisCandidate
    ) -> RefactorFinding:
        function_names = ", ".join(
            (function.function_name for function in case_candidate.functions[:6])
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
        shared_suffix = CLASS_NAME_ALGEBRA.longest_common_suffix(
            case_candidate.concrete_class_names
        )
        if (
            shared_suffix
            and len(shared_suffix) >= 6
            and not suggested_family_name.endswith(shared_suffix)
        ):
            suggested_family_name = f"{suggested_family_name}{shared_suffix}"
        elif not suggested_family_name.endswith(("Family", "Witness", "Variant")):
            suggested_family_name = f"{suggested_family_name}Family"
        return self.build_finding(
            (
                f"Functions {function_names} repeatedly recover `{case_candidate.subject_role}` across concrete classes {class_names}.{alias_summary}{existing_base_summary}"
            ),
            case_candidate.evidence,
            scaffold=(
                f"class {suggested_family_name}(ABC):\n    @property\n    @abstractmethod\n    def case_label(self) -> str: ...\n\n    def explain_case(self, context):\n        return None\n"
            ),
            codemod_patch=(
                f"# Type `{case_candidate.subject_role}` against one nominal ABC family instead of a concrete union surface.\n# Move repeated concrete `isinstance` recovery into abstract properties or case hooks on that family.\n# Keep only irreducible case-local residue in the concrete subclasses."
            ),
            metrics=DispatchCountMetrics(
                dispatch_site_count=len(case_candidate.functions),
                dispatch_axis=case_candidate.subject_role,
                literal_cases=case_candidate.concrete_class_names,
            ),
        )


class ImplicitSelfContractMixinDetector(
    ConfiguredCrossModuleCollectorCandidateDetector[ImplicitSelfContractMixinCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Concrete mixins should not hide consumer contracts behind `self`-casts",
        "The docs reserve mixins for orthogonal reusable concerns that participate in nominal MRO cleanly. When a concrete mixin erases `self` through `cast(..., self)` to reach consumer-owned fields, the mixin is carrying non-orthogonal family logic through a hidden contract instead of a declared base or policy.",
        "declared nominal base or policy row for the shared algorithm instead of a hidden mixin self-contract",
        "concrete mixin methods erase `self` through casts and depend on consumer-owned attributes across several subclasses",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS,
        _CLASS_FAMILY_REPEATED_METHOD_ROLES_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, mixin_candidate: ImplicitSelfContractMixinCandidate
    ) -> RefactorFinding:
        methods = ", ".join(mixin_candidate.method_names)
        consumers = ", ".join(mixin_candidate.consumer_class_names[:6])
        accessed_attributes = ", ".join(mixin_candidate.accessed_attribute_names[:6])
        cast_types = ", ".join(mixin_candidate.cast_type_names[:6])
        return self.build_finding(
            (
                f"`{mixin_candidate.mixin_name}` uses `cast(..., self)` ({cast_types}) in `{methods}` to reach consumer-owned attributes ({accessed_attributes}) across subclasses {consumers}."
            ),
            mixin_candidate.evidence,
            scaffold=(
                "class FamilyBase(ABC):\n    def run_shared_step(self): ...\n\nclass CasePolicy(ABC):\n    def run(self, request): ...\n"
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


class RepeatedGuardValidatorFamilyDetector(
    ConfiguredModuleCollectorCandidateDetector[RepeatedGuardValidatorFamilyCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Repeated guard validators should collapse into one case-policy authority",
        "When several sibling boolean helpers walk the same subject through fail-fast guards and case-local final checks, the algorithm skeleton is split across helper names instead of being owned by one nominal case policy or declarative rule family.",
        "single authoritative case-policy or rule-table validator",
        "same subject and subordinate view validated through repeated fail-fast sibling helpers",
        _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_AUTHORITATIVE_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_PARTIAL_VIEW_CLASS_FAMILY_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, family_candidate: RepeatedGuardValidatorFamilyCandidate
    ) -> RefactorFinding:
        function_names = ", ".join(
            (function.function_name for function in family_candidate.functions[:6])
        )
        shared_attrs = ", ".join(family_candidate.shared_attr_names[:6])
        alias_summary = (
            f" through `{family_candidate.alias_source_attr}`"
            if family_candidate.alias_source_attr is not None
            else ""
        )
        shared_helpers = ", ".join(family_candidate.shared_helper_call_names[:3])
        helper_summary = (
            f" Shared helper calls: {shared_helpers}." if shared_helpers else ""
        )
        return self.build_finding(
            (
                f"Boolean validators {function_names} each guard `{family_candidate.subject_param_name}`{alias_summary} "
                f"with the same fail-fast attribute checks over {shared_attrs}.{helper_summary}"
            ),
            family_candidate.evidence,
            scaffold=(
                "class ValidationCasePolicy(ABC):\n    def validation_error(self, subject):\n        child = self._subject_child(subject)\n        if not self._shared_preconditions(subject, child):\n            return self._shared_failure_message()\n        return self._case_specific_error(subject, child)\n\n    @abstractmethod\n    def _case_specific_error(self, subject, child): ..."
            ),
            codemod_patch=(
                "# Collapse these sibling boolean helpers into one authoritative case-policy family or one declarative rule table.\n# Keep shared fail-fast guards in one concrete validator method, and leave only case-specific predicates or handle sets per case."
            ),
        )


class AllMissingAxisPredicateDetector(
    ConfiguredModuleCollectorCandidateDetector[AllMissingAxisPredicateCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_CONTEXT,
        "All-missing axis predicates should be named axis authorities",
        "A raw conjunction of several `not axis` clauses is a derived predicate over a semantic axis bundle. Spelling that bundle inline makes the relation easy to fork and hard to audit. The normal form is a named tuple, policy, or context method that owns the axis set and lets the branch ask the derived question once.",
        "one named axis bundle or policy predicate deriving the all-missing condition",
        "three or more sibling axes are checked through an inline all-negative boolean conjunction before appending a missing signal",
        _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
    )

    def _finding_for_candidate(
        self, predicate_candidate: AllMissingAxisPredicateCandidate
    ) -> RefactorFinding:
        axis_names = ", ".join(predicate_candidate.predicate_names)
        return self.build_finding(
            (
                f"`{predicate_candidate.function_name}` checks all-missing axes "
                f"{axis_names} inline before appending `{predicate_candidate.signal_name}`."
            ),
            (predicate_candidate.evidence,),
            scaffold=(
                "rent_axes = (behavior_axis, abstract_axis, projection_axis, consumer_axis)\n"
                "if not any(rent_axes):\n"
                '    missing.append("derived_signal")'
            ),
            codemod_patch=(
                f"# Name the axis bundle in `{predicate_candidate.function_name}` before testing it.\n"
                f"# Replace the raw conjunction over {predicate_candidate.predicate_names} with `not any(axis_bundle)` "
                f"or a policy method that owns the same axes, then append `{predicate_candidate.signal_name}`."
            ),
        )


class RepeatedValidateShapeGuardFamilyDetector(IssueDetector):
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Repeated validate() shape guards should collapse into one validated-record authority",
        "Sibling nominal records repeat the same fail-fast shape and dimensional guards in `validate()` while differing only in field names or a small residue check. The docs treat that as duplicated contract authority that should move into one shared validated-record base, field-spec table, or mixin hook.",
        "single authoritative validated-record contract for repeated shape/ndim guards",
        "same nominal record family repeats fail-loud shape validation scaffolding",
        _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_AUTHORITATIVE_CAPABILITY_TAGS,
        _CLASS_FAMILY_METHOD_ROLE_NORMALIZED_AST_OBSERVATION_TAGS,
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
        return self.build_finding(
            (
                f"Validate methods {method_summary} repeat {shared_guard_count} shared shape/ndim guard forms."
            ),
            family_candidate.evidence,
            scaffold=(
                f"class ShapeValidatedRecord(ABC):\n    def validate(self):\n        for predicate, message in self._shape_guard_rules():\n            if predicate(self):\n                raise ValueError(message)\n        self._validate_residue()\n\n    @classmethod\n    @abstractmethod\n    def _shape_guard_rules(cls): ...\n\n    def _validate_residue(self):\n        return None{preview_suffix}"
            ),
            codemod_patch=(
                "# Collapse repeated `validate()` shape guards into one authoritative validated-record base or field-spec table.\n# Keep only the truly variable residue checks, messages, or field roster on each concrete record."
            ),
        )


class RepeatedResultAssemblyPipelineDetector(
    ConfiguredModuleCollectorCandidateDetector[RepeatedResultAssemblyPipelineCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Repeated result-assembly pipeline should collapse into one authoritative assembler",
        "Several owners repeat the same downstream result-assembly stages and differ only in the upstream source or projection that feeds the pipeline. The docs treat that as shared algorithm authority that should move into one template method or authoritative helper with one orthogonal source hook.",
        "single authoritative result-assembly pipeline with one source hook",
        "same staged assembly tail is repeated across sibling functions or methods",
        _SHARED_ALGORITHM_AUTHORITY_AUTHORITATIVE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
    )

    def _finding_for_candidate(
        self, pipeline_candidate: RepeatedResultAssemblyPipelineCandidate
    ) -> RefactorFinding:
        function_names = ", ".join(
            (function.qualname for function in pipeline_candidate.functions[:4])
        )
        stage_names = ", ".join(
            (stage.callee_name for stage in pipeline_candidate.shared_tail)
        )
        evidence = tuple(
            (function.evidence for function in pipeline_candidate.functions[:6])
        )
        return self.build_finding(
            (
                f"Functions {function_names} share the same result-assembly tail "
                f"{stage_names} and differ only in their leading source stages."
            ),
            evidence,
            scaffold=(
                "class ResultAssembler(ABC):\n    @abstractmethod\n    def supply_inputs(self, request): ...\n\n    def assemble(self, request):\n        supplied = self.supply_inputs(request)\n        # run the shared downstream assembly stages here\n        return result"
            ),
            codemod_patch=(
                "# Extract the shared assignment/return tail into one authoritative helper.\n# Leave only the source-supplier stage variant-specific."
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


@dataclass(frozen=True)
class _NormalFormScaffoldSpec:
    normal_form: str
    matcher_name: str
    method_name: str
    input_name: str
    step_base_name: str
    step_names: tuple[str, ...]

    def render(self) -> str:
        step_rows = "\n".join(
            (f"        {step_name}()," for step_name in self.step_names)
        )
        return f"class {self.step_base_name}(EffectStep, ABC):\n    normal_form = {self.normal_form!r}\n\n@dataclass(frozen=True)\nclass {self.matcher_name}:\n    steps: tuple[{self.step_base_name}, ...] = (\n{step_rows}\n    )\n\n    def {self.method_name}(self, {self.input_name}):\n        return Maybe.of({self.input_name}).bind_all(self.steps)"


_DEFAULT_NORMAL_FORM_SCAFFOLD = "class CandidateStep(EffectStep, ABC):\n    normal_form = 'typed_effect_carrier'\n\n@dataclass(frozen=True)\nclass CandidateMatcher:\n    steps: tuple[CandidateStep, ...] = (ExtractFirst(), ExtractSecond(), BuildWitness())\n\n    def build_candidate(self, source):\n        return Maybe.of(source).bind_all(self.steps)"
_NORMAL_FORM_SCAFFOLDS = {
    spec_name: _NormalFormScaffoldSpec(
        spec_name, matcher_name, method_name, input_name, step_base_name, steps
    )
    for spec_name, matcher_name, method_name, input_name, step_base_name, steps in (
        (
            "ast_shape_matcher",
            "AstShapeMatcher",
            "match_shape",
            "node",
            "AstShapeMatcherStep",
            (
                "ExpectCall",
                "ExpectSingleArgument",
                "ExpectNamedAstShape",
                "BuildWitness",
            ),
        ),
        (
            "transport_call_chain_matcher",
            "TransportChainMatcher",
            "match_transport_chain",
            "function",
            "TransportChainMatcherStep",
            (
                "SingleReturnCall",
                "CallChain",
                "TransportedValues",
                "BuildTransportWitness",
            ),
        ),
        (
            "comparison_guard_matcher",
            "ComparisonGuardMatcher",
            "match_comparison_guard",
            "test",
            "ComparisonGuardMatcherStep",
            ("SingleCompare", "EnumMemberPair", "BuildGuardPolicy"),
        ),
        (
            "loop_fold_matcher",
            "LoopFoldMatcher",
            "match_loop_fold",
            "body",
            "LoopFoldMatcherStep",
            (
                "ExpectAssignment",
                "ExpectLoop",
                "ExpectReturnedAccumulator",
                "BuildFoldWitness",
            ),
        ),
        (
            "statement_sequence_matcher",
            "StatementSequenceMatcher",
            "match_sequence",
            "function",
            "StatementSequenceMatcherStep",
            ("ExpectRoleSequence", "BuildWitness"),
        ),
    )
}


class FailSoftEffectPipelineDetector(
    ConfiguredModuleCollectorCandidateDetector[FailSoftEffectPipelineCandidate]
):
    finding_spec = finding_spec_template(
        PatternId.STAGED_ORCHESTRATION,
        "Fail-soft optional pipeline should use a typed effect carrier",
        "A function that repeatedly exits through `return None` is manually threading an optional effect through every extraction stage. The semantic-compressor normal form is a typed `Maybe`/`Result` carrier plus nominal matcher-step objects that own absence/provenance once instead of restating the guard at every stage.",
        "single typed effect carrier with nominal inherited matcher steps for optional extraction, validation, and provenance flow",
        "same fail-soft absence effect is manually re-threaded across one extraction pipeline",
        _SHARED_ALGORITHM_AUTHORITY_PROVENANCE_FAIL_LOUD_CONTRACTS_CAPABILITY_TAGS,
        _PREDICATE_CHAIN_DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    _DEFAULT_NORMAL_FORM_SCAFFOLD = _DEFAULT_NORMAL_FORM_SCAFFOLD
    _NORMAL_FORM_SCAFFOLDS: ClassVar[dict[str, _NormalFormScaffoldSpec]] = (
        _NORMAL_FORM_SCAFFOLDS
    )

    def _normal_form_scaffold(self, normal_form: str) -> str:
        spec = self._NORMAL_FORM_SCAFFOLDS.get(normal_form)
        return spec.render() if spec is not None else self._DEFAULT_NORMAL_FORM_SCAFFOLD

    def _finding_for_candidate(
        self, pipeline_candidate: FailSoftEffectPipelineCandidate
    ) -> RefactorFinding:
        binding_preview = ", ".join(pipeline_candidate.guarded_binding_names[:5])
        helper_preview = ", ".join(pipeline_candidate.helper_call_names[:5])
        binding_suffix = (
            f" over guarded bindings {binding_preview}" if binding_preview else ""
        )
        helper_suffix = (
            f" helper calls include {helper_preview}." if helper_preview else ""
        )
        return self.build_finding(
            (
                f"`{pipeline_candidate.function_name}` manually threads {pipeline_candidate.guard_count} "
                f"fail-soft guard stages{binding_suffix} before returning {pipeline_candidate.success_return_kind}; "
                f"normal form is `{pipeline_candidate.normal_form}`, family is "
                f"`{pipeline_candidate.pipeline_family}`, and owner should be "
                f"{pipeline_candidate.recommended_owner}."
                f"{helper_suffix}"
            ),
            (pipeline_candidate.evidence,),
            scaffold=self._normal_form_scaffold(pipeline_candidate.normal_form),
            codemod_patch=(
                f"# Collapse repeated `if value is None: return None` guard stages into `{pipeline_candidate.normal_form}`.\n"
                f"# {pipeline_candidate.refactor_action}.\n"
                "# Keep domain extraction semantics on nominal `EffectStep` subclasses; let the typed carrier own absence and provenance flow."
            ),
            metrics=OrchestrationMetrics(
                function_line_count=pipeline_candidate.line_count,
                branch_site_count=pipeline_candidate.guard_count,
                call_site_count=len(pipeline_candidate.helper_call_names),
                parameter_count=len(pipeline_candidate.guarded_binding_names),
                callee_family_count=max(1, len(pipeline_candidate.helper_call_names)),
            ),
        )


def _effect_step_payoff_scaffold(candidate: EffectStepAmortizationCandidate) -> str:
    normal_form_class = _camel_case(candidate.normal_form)
    return f"class {normal_form_class}Step(EffectStep, ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = 'step_id'\n    __skip_if_no_key__ = True\n    step_id: ClassVar[str | None] = None\n    registration_order: ClassVar[int] = 0\n\n@dataclass(frozen=True)\nclass {normal_form_class}Matcher:\n    steps: tuple[{normal_form_class}Step, ...]\n\n    def match(self, source):\n        return Maybe.of(source).bind_all(self.steps)"


class EffectStepAmortizationDetector(
    ConfiguredModuleCollectorCandidateDetector[EffectStepAmortizationCandidate]
):
    finding_spec = finding_spec_template(
        PatternId.STAGED_ORCHESTRATION,
        "Manual AST matcher should amortize EffectStep infrastructure",
        "A helper that repeatedly performs AST type/cardinality checks and exits through `return None` is paying the cognitive cost of an effect pipeline without reusing the nominal `EffectStep` carrier. The infrastructure pays rent when these guard atoms become registered matcher-step objects that can be shared, ordered, tested, and composed.",
        "reusable nominal EffectStep family for recurring AST type, cardinality, and optional-exit guards",
        "same optional AST matcher mechanics are hand-expanded inside one helper",
        _SHARED_ALGORITHM_AUTHORITY_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _PREDICATE_CHAIN_NORMALIZED_AST_DATAFLOW_ROOT_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, payoff_candidate: EffectStepAmortizationCandidate
    ) -> RefactorFinding:
        ast_types = ", ".join(payoff_candidate.ast_type_names[:5]) or "no ast types"
        helpers = (
            ", ".join(payoff_candidate.semantic_helper_names[:5])
            or "no semantic helpers"
        )
        return self.build_finding(
            (
                f"`{payoff_candidate.function_name}` has payoff score "
                f"{payoff_candidate.payoff_score}: {payoff_candidate.none_return_count} optional exits, "
                f"{payoff_candidate.ast_type_guard_count} AST type guards over {ast_types}, "
                f"{payoff_candidate.cardinality_guard_count} cardinality guards, and "
                f"{payoff_candidate.semantic_helper_count} semantic helper calls ({helpers}); "
                f"normal form is `{payoff_candidate.normal_form}` with "
                f"manual object score {payoff_candidate.payoff_score}, generated budget "
                f"{payoff_candidate.generated_object_budget}, and net object savings "
                f"{payoff_candidate.net_object_savings}; semantic description length "
                f"{payoff_candidate.description_length_before} -> "
                f"{payoff_candidate.description_length_after} with certified savings "
                f"{payoff_candidate.description_length_savings}."
            ),
            (payoff_candidate.evidence,),
            scaffold=_effect_step_payoff_scaffold(payoff_candidate),
            compression_certificate=payoff_candidate.compression_certificate,
            codemod_patch=(
                "# Delete the manual guard chain only when the computed algebraic budget is positive.\n# Extract repeated AST guard atoms into nominal `EffectStep` subclasses or generated step declarations.\n# Register ordered steps with `AutoRegisterMeta`, then route the helper through `Maybe.of(source).bind_all(steps)`.\n# Keep only domain-specific witness construction outside the shared matcher pipeline."
            ),
            metrics=OrchestrationMetrics(
                function_line_count=payoff_candidate.line_count,
                branch_site_count=payoff_candidate.none_return_count,
                call_site_count=payoff_candidate.semantic_helper_count,
                parameter_count=len(payoff_candidate.ast_type_names),
                callee_family_count=max(1, len(payoff_candidate.semantic_helper_names)),
            ),
        )


declare_candidate_rule_detector(
    EffectStepImplementationLeakCandidate,
    high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "EffectStep leaf should declare hooks instead of owning apply",
        "Concrete effect-step leaves should carry semantic residue as attributes/properties and small hooks. When a leaf owns raw optional exits, AST type checks, or cardinality checks inside `apply()` or a bulky hook, the ABC is not doing enough of the work and the monadic infrastructure is not compressing semantics.",
        "template-method EffectStep base that owns optional flow, type narrowing, and guard sequencing",
        "concrete EffectStep leaf repeats mechanics that belong in an ABC/template base",
        _FAIL_LOUD_CONTRACTS_SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _PREDICATE_CHAIN_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda leak: f"`{leak.class_name}.{leak.method_name}` owns {leak.raw_guard_count} raw guard mechanics and {leak.none_return_count} optional exits; move the algorithm into `{leak.suggested_base_name}` and leave only attrs/properties plus hooks on the leaf.",
    scaffold=lambda leak: f"class {leak.class_name}({leak.suggested_base_name}):\n    step_id = 'semantic_step'\n    registration_order = 10\n    # declare class attrs/properties here\n\n    def accepts(self, value): ...\n    def project(self, value): ...",
    codemod_patch=lambda leak: "# Delete the concrete mechanics-heavy leaf method.\n# Move optional flow/type narrowing/cardinality mechanics to the ABC/template base.\n# Keep the implementation class declarative: attrs, properties, and the smallest semantic hooks.",
    metrics=lambda leak: OrchestrationMetrics(
        function_line_count=0,
        branch_site_count=leak.none_return_count,
        call_site_count=leak.raw_guard_count,
        parameter_count=0,
        callee_family_count=1,
    ),
    candidate_collector=_effect_step_implementation_leak_candidates,
)


class UnderAmortizedInfrastructureDetector(
    CrossModuleCollectorCandidateDetector[UnderAmortizedInfrastructureCandidate]
):
    finding_spec = finding_spec_template(
        PatternId.STAGED_ORCHESTRATION,
        "Matcher infrastructure should pay rent through fanout",
        "A shared matcher/effect infrastructure module should earn its declarations through repeated external use. When a public helper or carrier has only one external consumer and is not support for a broadly reused declaration, the abstraction is expanding the surface area faster than it compresses manual code.",
        "public matcher infrastructure whose declaration cost is amortized by multiple consumers",
        "effect/matcher module public surface has single-consumer declarations",
        _SHARED_ALGORITHM_AUTHORITY_UNIT_RATE_COHERENCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, under_amortized: UnderAmortizedInfrastructureCandidate
    ) -> RefactorFinding:
        declaration_preview = ", ".join(under_amortized.declaration_names[:8])
        consumer_preview = ", ".join(under_amortized.consumer_symbols[:4])
        support_suffix = (
            f" Single-consumer support declarations: {', '.join(under_amortized.support_names[:8])}."
            if under_amortized.support_names
            else ""
        )
        return self.build_finding(
            (
                f"{under_amortized.file_path} exposes single-consumer matcher infrastructure "
                f"{declaration_preview}; consumers: {consumer_preview}.{support_suffix}"
            ),
            (under_amortized.evidence,),
            scaffold=(
                "# Either inline the single-consumer declaration into its only consumer, or merge it into an already-amortized primitive.\n# Keep new public matcher infrastructure only when fanout shows more than one external consumer."
            ),
            codemod_patch=(
                "# Collapse the single-consumer public surface before adding more matcher machinery.\n# If the declaration represents real reusable semantics, route at least two consumers through it."
            ),
            metrics=OrchestrationMetrics(
                function_line_count=0,
                branch_site_count=len(under_amortized.declaration_names),
                call_site_count=len(under_amortized.consumer_symbols),
                parameter_count=len(under_amortized.support_names),
                callee_family_count=1,
            ),
        )


class PublicBareSupportFunctionDetector(
    CrossModuleCollectorCandidateDetector[PublicBareSupportFunctionCandidate]
):
    finding_spec = finding_spec_template(
        PatternId.NOMINAL_INTERFACE_WITNESS,
        "Public bare support function in private module should become nominal",
        "A public snake_case function inside an underscore-prefixed support module is usually a promoted helper, not a real public contract. If the behavior is reusable, the semantic owner should be an ABC method, projection/authority object, catalog, descriptor, or registered step; otherwise the function should stay private and local to its only owner.",
        "nominal owner for reusable support behavior rather than public helper surface",
        "private support module exposes public bare functions without a nominal owner",
        _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_AUTHORITATIVE_CAPABILITY_TAGS,
        _METHOD_ROLE_NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, support_candidate: PublicBareSupportFunctionCandidate
    ) -> RefactorFinding:
        function_preview = ", ".join(support_candidate.function_names[:10])
        return self.build_finding(
            (
                f"{support_candidate.file_path} exposes public bare support functions "
                f"{function_preview} from private module role `{support_candidate.module_role}`; "
                f"semantic family is `{support_candidate.semantic_family}` and owner should be "
                f"`{support_candidate.recommended_owner}`; "
                f"external references: {support_candidate.external_reference_count}."
            ),
            (support_candidate.evidence,),
            scaffold=(
                f"class {support_candidate.recommended_owner}:\n"
                "    def project(self, source): ...\n\n"
                "# Or move the behavior onto the existing ABC/catalog/descriptor/EffectStep that owns the axis."
            ),
            codemod_patch=(
                f"# Replace public module-level helper exports in `{support_candidate.semantic_family}` with nominal owner `{support_candidate.recommended_owner}`.\n"
                "# Keep tiny single-owner residue private; route reusable behavior through an ABC method, authority object, descriptor, catalog, or AutoRegisterMeta-backed step."
            ),
            metrics=OrchestrationMetrics(
                function_line_count=0,
                branch_site_count=0,
                call_site_count=support_candidate.external_reference_count,
                parameter_count=len(support_candidate.function_names),
                callee_family_count=1,
            ),
        )


declare_candidate_rule_detector(
    DetectorBackendPayoffGuardCandidate,
    high_confidence_spec(
        PatternId.STAGED_ORCHESTRATION,
        "Detector abstraction advice should prove backend LOC payoff",
        "A detector that recommends helpers, wrappers, template bases, registries, or other abstraction machinery must make the rent check explicit: the proposed abstraction should reduce semantic description length, reduce backend LOC, or carry fanout/amortization evidence that explains why the added infrastructure pays rent.",
        "detector advice includes a compression certificate or backend LOC/fanout budget with net reduction action",
        "detector recommends abstraction without proving the backend LOC payoff",
        _SHARED_ALGORITHM_AUTHORITY_UNIT_RATE_COHERENCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda detector: f"`{detector.qualname}` uses abstraction terms {detector.abstraction_terms} but is missing payoff guard(s) {detector.missing_guard_names}; detector refactors must carry a `CompressionCertificate`, cut backend LOC, or expose fanout.",
    scaffold=lambda detector: "# Add `compression_certificate=` or structured `metrics=` to the detector declaration.\n# Candidate evidence should expose manual cost, replacement grammar cost, residuals, margin, or fanout.",
    codemod_patch=lambda detector: "# Require `compression_certificate=` for MDL-style abstraction advice when possible.\n# Otherwise require structured backend/fanout metrics, or inline/delete under-amortized machinery.",
    metrics=lambda detector: OrchestrationMetrics(
        function_line_count=detector.declaration_line_count,
        branch_site_count=len(detector.missing_guard_names),
        call_site_count=1,
        parameter_count=len(detector.abstraction_terms),
        callee_family_count=1,
    ),
    detector_priority=-18,
    candidate_collector=_detector_backend_payoff_guard_candidates,
)


declare_candidate_rule_detector(
    CandidateCollectorBoilerplateCandidate,
    high_confidence_spec(
        PatternId.STAGED_ORCHESTRATION,
        "Candidate detector should declare collector strategy",
        "Detector classes repeatedly implement `_candidate_items()` as a one-line forwarding method. That is boilerplate control flow: the detector identity and finding rendering are semantic, while candidate collection is a typed class-level strategy that can be inherited.",
        "typed metaprogrammed detector base that derives candidate collection from a declared strategy",
        "detector class repeats collector forwarding method instead of declaring a collector",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_UNIT_RATE_COHERENCE_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda collector: f"`{collector.class_name}.{collector.method_name}` only forwards to `{collector.collector_name}`; inherit `{collector.recommended_base_name}` and declare `candidate_collector` instead.",
    scaffold=lambda collector: f"class {collector.class_name}({collector.recommended_base_name}):\n    candidate_collector = {collector.collector_name}\n",
    codemod_patch=lambda collector: f"# Delete the forwarding `_candidate_items()` method.\n# Change the detector base to `{collector.recommended_base_name}` and assign `candidate_collector = {collector.collector_name}`.",
    metrics=lambda collector: OrchestrationMetrics(
        function_line_count=0,
        branch_site_count=1,
        call_site_count=1,
        parameter_count=2 if collector.uses_config else 1,
        callee_family_count=1,
    ),
    detector_priority=-19,
    candidate_collector=_candidate_collector_boilerplate_candidates,
)


declare_candidate_rule_detector(
    TypedCandidateCastBoilerplateCandidate,
    high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Candidate template method should receive typed candidates directly",
        "Detector classes repeatedly accept `candidate: object`, immediately cast it to a nominal candidate type, and then never use the object-typed parameter again. That cast belongs in the generic detector base contract: the implementation hook should receive the typed candidate directly.",
        "generic typed candidate detector base with no per-detector cast prelude",
        "candidate-rendering template method starts with a local cast of its only payload parameter",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: f"`{candidate.class_name}.{candidate.method_name}` casts `{candidate.parameter_name}` to `{candidate.candidate_type_name}` before doing real work; parameterize `{candidate.detector_base_name}` and receive `{candidate.local_name}` as that type.",
    scaffold=lambda candidate: f"class {candidate.class_name}({candidate.detector_base_name}[{candidate.candidate_type_name}]):\n    def {candidate.method_name}(self, {candidate.local_name}: {candidate.candidate_type_name}) -> RefactorFinding:\n        ...",
    codemod_patch=lambda candidate: f"# Change the detector base to `{candidate.detector_base_name}[{candidate.candidate_type_name}]`.\n# Rename the hook argument from `{candidate.parameter_name}` to `{candidate.local_name}` and delete the local `cast(...)` prelude.",
    metrics=lambda candidate: _SINGLE_TEMPLATE_CALL_METRICS,
    detector_priority=-18,
    candidate_collector=_typed_candidate_cast_boilerplate_candidates,
)


declare_candidate_rule_detector(
    DeclarativeDetectorClassCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Metadata-only detector class should be declared through detector algebra",
        "A detector class whose body only assigns finding metadata and a renderer is not carrying implementation behavior. Its class shell is derivable from the candidate type, detector base, registry key, and declaration line.",
        "one detector-declaration algebra that derives metadata-only detector classes",
        "detector class repeats a nominal class shell around only declarative assignments",
        _AUTHORITATIVE_NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS,
        _CLASS_FAMILY_NORMALIZED_AST_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: f"`{candidate.class_name}` is a {candidate.line_count}-line metadata-only detector over `{candidate.candidate_type_name}` with assignments {candidate.assignment_names}.",
    scaffold=lambda candidate: f"declare_module_detector({candidate.candidate_type_name}, finding_spec, finding_renderer, detector_base={candidate.base_name})",
    codemod_patch=lambda candidate: f"# Replace `{candidate.class_name}` with `declare_module_detector(...)`.\n# Keep only true detector-specific values: spec, renderer, optional collector, base, and priority.",
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=candidate.line_count,
        mapping_name=candidate.class_name,
        field_names=candidate.assignment_names,
        source_name=candidate.base_name,
    ),
    detector_priority=-17,
    candidate_collector=_declarative_detector_class_candidates,
)


declare_candidate_rule_detector(
    StaticTypedObservationDetectorCandidate,
    high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Static observation detector should derive from typed observation algebra",
        "A static detector whose evidence method only collects one typed observation family and maps its line/symbol payload into `SourceLocation` is repeating the same module-observation algorithm. The detector should declare the observation family, item type, evidence threshold, and summary template while the ABC owns collection and evidence projection.",
        "typed observation detector algebra with one shared collection/projection algorithm",
        "detector class shell repeats collect/map/summary mechanics for one observation family",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_UNIT_RATE_COHERENCE_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: f"`{candidate.class_name}` repeats a {candidate.line_count}-line static observation shell over `{candidate.observation_family_name}` / `{candidate.observation_type_name}`.",
    scaffold=lambda candidate: f'declare_typed_observation_detector(\n    "{candidate.class_name}",\n    finding_spec,\n    {candidate.observation_family_name},\n    {candidate.observation_type_name},\n    summary_template,\n)',
    codemod_patch=lambda candidate: f"# Replace `{candidate.class_name}` with `declare_typed_observation_detector(...)`.\n# Keep detector-specific semantics as declarations: finding spec, observation family/type, minimum evidence, and summary template.",
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=candidate.line_count,
        mapping_name=candidate.class_name,
        field_names=(
            "finding_spec",
            "observation_family",
            "observation_type",
            "minimum_evidence_count",
            "summary_template",
        ),
        source_name=candidate.observation_family_name,
    ),
    detector_priority=-16,
    candidate_collector=_static_typed_observation_detector_candidates,
)


declare_candidate_rule_detector(
    InlineCandidateRendererDeclarationCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Inline renderer declaration should fold into detector algebra",
        "A detector declaration that embeds `CandidateFindingRenderer[...]` repeats the same rendering-object construction beside every candidate type. The detector algebra already knows the candidate type, finding spec, collector, and priority; it should derive the renderer object from typed summary/evidence/scaffold/patch/metrics hooks.",
        "one detector declaration algebra derives the renderer object and detector class",
        "module detector declaration manually embeds candidate renderer construction",
        _AUTHORITATIVE_NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda renderer: f"`{renderer.qualname}` embeds renderer keywords {renderer.renderer_keyword_names} across {renderer.line_count} lines; declare the detector rule directly and derive the renderer.",
    scaffold=lambda renderer: "declare_candidate_rule_detector(\n    Candidate,\n    finding_spec,\n    summary=lambda candidate: ...,\n)",
    codemod_patch=lambda renderer: "# Replace the nested `CandidateFindingRenderer[...]` argument with `declare_candidate_rule_detector(...)` keyword hooks. Omit `evidence=` when it is exactly `lambda candidate: (candidate.evidence,)`.",
    metrics=lambda renderer: MappingMetrics.from_field_names(
        mapping_site_count=renderer.line_count,
        mapping_name=renderer.candidate_type_name,
        field_names=(
            *renderer.renderer_keyword_names,
            *renderer.detector_keyword_names,
        ),
        source_name="CandidateFindingRenderer",
    ),
    detector_priority=-15,
    candidate_collector=_inline_candidate_renderer_declaration_candidates,
)


declare_candidate_rule_detector(
    NamedFunctionCollectorBoilerplateCandidate,
    high_confidence_certified_spec(
        PatternId.STAGED_ORCHESTRATION,
        "Named-function candidate collectors should share traversal algebra",
        "A candidate collector that manually initializes `candidates`, walks `_iter_named_functions(module)`, appends candidate records, and returns the accumulator is repeating traversal and accumulator mechanics. The collector should delegate named-function traversal to one typed query helper and keep only the per-function projection as semantic residue.",
        "one named-function collector algebra owns traversal and accumulation",
        "candidate collector repeats named-function traversal with manual append accumulator",
        _SHARED_ALGORITHM_AUTHORITY_UNIT_RATE_COHERENCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda collector: f"`{collector.function_name}` manually accumulates {collector.candidate_type_names} over `_iter_named_functions(module)` across {collector.line_count} lines.",
    scaffold=lambda collector: "def _candidate_for_function(module, qualname, function):\n    yield Candidate(...)\n\ndef _candidates(module):\n    return _collect_named_function_candidates(module, _candidate_for_function)",
    codemod_patch=lambda collector: "# Delete the manual `_iter_named_functions(module)` traversal and route collection through `_collect_named_function_candidates(...)`; keep only the per-function projection hook.",
    metrics=lambda collector: OrchestrationMetrics(
        function_line_count=collector.line_count,
        branch_site_count=collector.append_count,
        call_site_count=1,
        parameter_count=len(collector.candidate_type_names),
        callee_family_count=1,
    ),
    detector_priority=-14,
    candidate_collector=_named_function_collector_boilerplate_candidates,
)


declare_candidate_rule_detector(
    IdentityKeywordForwardingShellCandidate,
    high_confidence_certified_spec(
        PatternId.LOCAL_VALUE_AUTHORITY,
        "Identity keyword forwarding shell should collapse into the semantic authority",
        "A function whose complete body is `return Authority(field=field, ...)` and whose forwarded keyword names exactly match its own parameters has no independent invariant, policy, or provenance boundary. The stable object is the callee authority or a typed request record, not the transport shell.",
        "direct authority call or typed request object instead of a same-name keyword relay",
        "single-return wrapper forwards every parameter as an identically named keyword",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda shell: f"`{shell.function_name}` only forwards {shell.forwarded_keyword_names} to `{shell.callee_name}` with identical keyword names.",
    scaffold=lambda shell: (
        f"# Delete `{shell.function_name}` and call `{shell.callee_name}` directly.\n"
        "# If the parameter family is semantically real, replace the parameter list with one typed request record."
    ),
    codemod_patch=lambda shell: (
        f"# Inline `{shell.function_name}` at call sites and remove the wrapper.\n"
        "# Preserve only invariants that are not already owned by the callee authority."
    ),
    compression_certificate=lambda shell: CompressionCertificate.from_object_family(
        manual_object_count=max(
            shell.line_count, len(shell.forwarded_keyword_names) + 1
        ),
        replacement_shape=ObjectFamilyShape(shared_objects=("callee_authority",)),
        semantic_axes=shell.forwarded_keyword_names,
    ),
    metrics=lambda shell: ParameterThreadMetrics(
        function_count=1,
        shared_parameter_count=len(shell.forwarded_keyword_names),
        shared_parameter_names=shell.forwarded_keyword_names,
    ),
    detector_priority=-13,
    candidate_collector=_identity_keyword_forwarding_shell_candidates,
)


declare_candidate_rule_detector(
    OptionalKeywordBagAssemblyCandidate,
    high_confidence_certified_spec(
        PatternId.LOCAL_VALUE_AUTHORITY,
        "Optional keyword bag assembly should become a named call variant",
        "A function that initializes a temporary dict, guards several optional parameters with `is not None`, copies same-name values into that dict, and unpacks it into a call is encoding a constructor/call variant as branch mechanics. The stable semantic object is a named variant factory, policy, or direct call surface that owns those optional coordinates.",
        "named call variant or factory instead of optional keyword bag mutation",
        "empty dict plus repeated non-None guards feeds a call through **kwargs",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda bag: (
        f"`{bag.function_name}` builds `{bag.bag_name}` from optional parameters "
        f"{bag.parameter_names} and unpacks it into `{bag.call_name}`."
    ),
    scaffold=lambda bag: (
        f"# Replace `{bag.bag_name}` with a named factory/variant for `{bag.call_name}`.\n"
        "# Move optional-coordinate policy into the factory type or call the concrete constructor directly."
    ),
    codemod_patch=lambda bag: (
        f"# Delete the mutable `{bag.bag_name}` assembly and its repeated `is not None` branches.\n"
        "# Use a named spec factory/variant whose constructor signature exposes only valid coordinates."
    ),
    compression_certificate=lambda bag: CompressionCertificate.from_object_family(
        manual_object_count=bag.line_count,
        replacement_shape=ObjectFamilyShape.from_roles(
            ("call_variant_factory",),
            axis=bag.target_keyword_names,
        ),
        semantic_axes=bag.target_keyword_names,
    ),
    metrics=lambda bag: ParameterThreadMetrics(
        function_count=1,
        shared_parameter_count=len(bag.parameter_names),
        shared_parameter_names=bag.parameter_names,
    ),
    detector_priority=-13,
    candidate_collector=_optional_keyword_bag_assembly_candidates,
)


declare_candidate_rule_detector(
    TupleIndexSemanticOpacityCandidate,
    high_confidence_certified_spec(
        PatternId.LOCAL_VALUE_AUTHORITY,
        "Carrier tuple context should become a named semantic record",
        "A typed carrier pipeline that accesses context as `pair[0]`, `pair[1]`, or nested numeric tuple paths has collapsed the control-flow smell but introduced a positional data smell. The semantic-compressor normal form is a named product record, authority-owned context object, or step result type whose field names carry the invariant.",
        "named effect context record instead of positional tuple plumbing",
        "effect pipeline stores semantic context in numeric tuple indexes",
        _UNIT_RATE_COHERENCE_AUTHORITATIVE_PROVENANCE_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: (
        f"`{candidate.function_name}` uses positional tuple paths "
        f"{candidate.index_expressions} inside carrier pipeline calls "
        f"{candidate.carrier_call_names}."
    ),
    scaffold=lambda candidate: (
        "from dataclasses import dataclass\n\n"
        "@dataclass(frozen=True)\n"
        "class PipelineContext:\n"
        "    source: Source\n"
        "    projection: Projection\n"
        "# Replace `pair[0]`/`pair[1]` with named fields derived once by the carrier stage."
    ),
    codemod_patch=lambda candidate: (
        "# Introduce a named product record or authority-owned context for the carrier stage.\n"
        "# Replace numeric tuple paths with named fields; keep the carrier, but stop encoding semantics by position."
    ),
    compression_certificate=lambda candidate: CompressionCertificate.from_object_family(
        manual_object_count=candidate.nested_index_count
        + len(candidate.index_expressions),
        replacement_shape=ObjectFamilyShape(shared_objects=("named_effect_context",)),
        semantic_axes=candidate.index_expressions,
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=candidate.nested_index_count,
        mapping_name=candidate.function_name,
        field_names=candidate.index_expressions,
        source_name="carrier_tuple_context",
    ),
    detector_priority=-13,
    candidate_collector=_tuple_index_semantic_opacity_candidates,
)


declare_candidate_rule_detector(
    OptionalParameterBranchCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Optional parameter branch should become a nominal variant hook",
        "A parameter annotated as optional, tested with `is None`, and also observed through methods/properties is encoding a behavior axis through absence. If the branch changes behavior, the semantic-compressor normal form is an ABC or nominal strategy family whose concrete variants own the absent/present behavior instead of a nullable signature plus local branch.",
        "ABC or nominal strategy variants for the absent/present case",
        "function signature admits `None`, branches on that parameter, and calls/reads through the same parameter as an object",
        _NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda branch: f"`{branch.function_name}` accepts `{branch.parameter_name}: {branch.annotation_text}`, branches on `{branch.parameter_name} is None` {branch.none_check_count} time(s), and observes attributes {branch.observed_attribute_names}.",
    scaffold=lambda branch: (
        f"class {branch.parameter_name.title().replace('_', '')}Policy(ABC):\n"
        "    @abstractmethod\n"
        "    def apply(self, context): ...\n\n"
        "# Replace the nullable parameter with a concrete policy/variant object."
    ),
    codemod_patch=lambda branch: (
        f"# Split `{branch.parameter_name}` absence/presence into named variants.\n"
        "# Move the None branch into a default/null-object implementation and delete the local optional check."
    ),
    compression_certificate=lambda branch: CompressionCertificate.from_object_family(
        manual_object_count=max(branch.line_count, branch.none_check_count + 2),
        replacement_shape=ObjectFamilyShape(
            shared_objects=("optional_axis_abc",),
            per_axis_objects=("variant_hook",),
        ),
        semantic_axes=(branch.parameter_name,),
        residual_object_count=2,
    ),
    metrics=lambda branch: OrchestrationMetrics(
        function_line_count=branch.line_count,
        branch_site_count=branch.none_check_count,
        call_site_count=0,
        parameter_count=1,
        callee_family_count=2,
    ),
    detector_priority=-12,
    candidate_collector=_optional_parameter_branch_candidates,
)


declare_candidate_rule_detector(
    BareFunctionMethodFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Bare function family should move behind a nominal owner",
        "A cohort of module-level functions sharing the same first semantic parameter and name axis is acting like methods without an owner. The shared parameter is the missing nominal authority; push common behavior into an ABC/template object and leave the top-level surface as thin orchestration only when it owns an actual boundary.",
        "nominal method family authority instead of loose subject-parameter functions",
        "module-level function cohort shares a subject parameter and semantic name axis",
        _AUTHORITATIVE_NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda family: (
        f"`{family.file_path}` has bare functions {family.function_names} sharing "
        f"first parameter `{family.owner_parameter_name}` and "
        f"{family.shared_axis_name} axis `{family.shared_axis_value}` while reading "
        f"owner attributes {family.owner_attribute_names}."
    ),
    scaffold=lambda family: (
        "class SubjectMethodFamily(ABC):\n"
        f"    # Own `{family.owner_parameter_name}` here; keep each function's "
        "irreducible behavior as a hook.\n"
        "    @abstractmethod\n"
        "    def run(self): ...\n"
    ),
    codemod_patch=lambda family: (
        "# Move the shared subject-parameter function family behind a nominal "
        "ABC/template authority.\n"
        "# Convert each bare function into a method, hook, or strategy case on "
        f"the owner of `{family.owner_parameter_name}`."
    ),
    compression_certificate=lambda family: family.compression_certificate,
    metrics=lambda family: OrchestrationMetrics(
        function_line_count=family.line_count,
        branch_site_count=0,
        call_site_count=len(family.function_names),
        parameter_count=1,
        callee_family_count=len(family.function_names),
    ),
    detector_priority=-12,
    candidate_collector=_bare_function_method_family_candidates,
)


declare_candidate_rule_detector(
    LatentNominalFunctionFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Scattered function cohort should become a nominal owner family",
        "A cohort of module-level functions shares the same semantic first parameter and repeatedly reads the same owner attributes, but its naming does not expose a simple prefix/suffix axis. The code is still acting like a latent object: the shared parameter should become a nominal owner/ABC and the functions should become methods, hooks, or strategy operations.",
        "nominal owner ABC with operation hooks derived from a scattered function cohort",
        "module-level functions share an owner parameter and attribute surface without a named owner",
        _AUTHORITATIVE_NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda family: (
        f"`{family.file_path}` has scattered functions {family.function_names} "
        f"sharing first parameter `{family.owner_parameter_name}` and owner "
        f"attributes {family.owner_attribute_names}; recover the latent nominal owner"
        + (
            f" with consumer fanout {family.consumer_symbols}."
            if family.consumer_symbols
            else "."
        )
    ),
    scaffold=lambda family: (
        "class LatentOwnerFamily(ABC):\n"
        f"    # Own `{family.owner_parameter_name}` and expose shared attributes "
        f"{family.owner_attribute_names} through a nominal contract.\n"
        "    @abstractmethod\n"
        "    def run_operation(self): ..."
    ),
    codemod_patch=lambda family: (
        f"# Move scattered functions {family.function_names} behind a nominal owner "
        f"for `{family.owner_parameter_name}`.\n"
        "# Convert each operation into a method/hook/strategy case; keep top-level "
        "functions only as compatibility facades if they are public API."
    ),
    compression_certificate=lambda family: family.compression_certificate,
    metrics=lambda family: OrchestrationMetrics(
        function_line_count=family.line_count,
        branch_site_count=0,
        call_site_count=len(family.function_names) + len(family.consumer_symbols),
        parameter_count=1,
        callee_family_count=len(family.function_names),
    ),
    detector_priority=-12,
    candidate_collector=_latent_nominal_function_family_candidates,
)


declare_candidate_rule_detector(
    AstStreamCollectorBoilerplateCandidate,
    high_confidence_certified_spec(
        PatternId.STAGED_ORCHESTRATION,
        "AST stream candidate collectors should share traversal algebra",
        "A candidate collector that manually creates a list accumulator, walks an AST stream such as `ast.walk(...)` or `_walk_nodes(...)`, appends candidate records, and returns the accumulator is repeating stream traversal and accumulation mechanics. The collector should delegate typed AST stream traversal to one query helper and keep only node-level projection semantics.",
        "one typed AST stream collector algebra owns traversal and accumulation",
        "candidate collector repeats AST stream traversal with manual append accumulator",
        _SHARED_ALGORITHM_AUTHORITY_UNIT_RATE_COHERENCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda collector: f"`{collector.function_name}` manually accumulates {collector.candidate_type_names} from {collector.stream_call_names} via `{collector.accumulator_name}` across {collector.line_count} lines.",
    scaffold=lambda collector: "def _candidate_for_node(module, node):\n    yield Candidate(...)\n\ndef _candidates(module):\n    return _collect_ast_node_candidates(module, module.module, ast.NodeType, _candidate_for_node)",
    codemod_patch=lambda collector: "# Delete the manual AST traversal and route collection through `_collect_ast_node_candidates(...)`; keep only the typed node projection hook.",
    metrics=lambda collector: OrchestrationMetrics(
        function_line_count=collector.line_count,
        branch_site_count=collector.append_count,
        call_site_count=len(collector.stream_call_names),
        parameter_count=len(collector.candidate_type_names),
        callee_family_count=1,
    ),
    detector_priority=-13,
    candidate_collector=_ast_stream_collector_boilerplate_candidates,
)


class FindingSpecDefaultFieldBoilerplateDetector(
    ModuleCollectorCandidateDetector[FindingSpecDefaultFieldCandidate]
):
    candidate_collector = _finding_spec_default_field_candidates
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "FindingSpec semantic defaults should be constructor-derived",
        "FindingSpec constructors already encode confidence and certification defaults. Restating those semantic fields in every detector is declaration boilerplate; the constructor should carry the shared semantic tier and leave only true local residue.",
        "constructor-level semantic spec defaults with no repeated confidence/certification payload",
        "FindingSpec call repeats semantic default keywords that can be derived from its constructor",
        _AUTHORITATIVE_SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, field_candidate: FindingSpecDefaultFieldCandidate
    ) -> RefactorFinding:
        keyword_summary = ", ".join(
            (
                f"{name}={value}"
                for name, value in zip(
                    field_candidate.redundant_keyword_names,
                    field_candidate.redundant_keyword_values,
                    strict=True,
                )
            )
        )
        constructor_note = (
            f" and use `{field_candidate.recommended_constructor_name}`"
            if field_candidate.recommended_constructor_name
            != field_candidate.constructor_name
            else ""
        )
        return self.build_finding(
            (
                f"`{field_candidate.constructor_name}` restates derived semantic defaults "
                f"{keyword_summary}{constructor_note}."
            ),
            (field_candidate.evidence,),
            scaffold=(
                f"{field_candidate.recommended_constructor_name}(\n    pattern_id=...,\n    title=...,\n    ...\n)"
            ),
            codemod_patch=(
                f"# Replace `{field_candidate.constructor_name}` with "
                f"`{field_candidate.recommended_constructor_name}` where needed.\n"
                f"# Delete redundant semantic keywords: {', '.join(field_candidate.redundant_keyword_names)}."
            ),
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(field_candidate.redundant_keyword_names),
                mapping_name=field_candidate.constructor_name,
                field_names=field_candidate.redundant_keyword_names,
            ),
        )


declare_candidate_rule_detector(
    ClassMethodLineWitnessCandidate,
    high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Detector finding builder should derive detector_id",
        "Concrete detectors repeatedly call `self.finding_spec.build(self.detector_id, ...)`. The detector id is instance-owned template context, not per-finding payload; a shared `build_finding(...)` hook should inject it once.",
        "typed detector template method that injects detector identity into finding construction",
        "finding renderer manually passes detector-owned identity into its own spec builder",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_AUTHORITATIVE_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: f"`{candidate.symbol}` calls `self.finding_spec.build(self.detector_id, ...)`; `build_finding(...)` can derive the detector id from the instance.",
    scaffold=lambda candidate: "return self.build_finding(\n    summary,\n    evidence,\n    ...\n)",
    codemod_patch=lambda candidate: "# Replace `self.finding_spec.build(` with `self.build_finding(`.\n# Delete the leading `self.detector_id,` argument.",
    metrics=lambda candidate: _SINGLE_TEMPLATE_CALL_METRICS,
    detector_name="FindingSpecBuildBoilerplateDetector",
    candidate_collector=_finding_spec_build_boilerplate_candidates,
)


class DirectBuildFindingRendererDetector(
    ModuleCollectorCandidateDetector[DirectBuildFindingRendererCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Direct build_finding renderer should be a typed renderer value",
        "A `_finding_for_candidate` method whose entire body is `return self.build_finding(...)` does not own control flow. It is a data renderer over one candidate type, so the candidate-to-finding algorithm should live once in the ABC and the detector should supply a typed renderer object.",
        "typed candidate finding renderer reused by detector ABC machinery",
        "detector method is only a build_finding payload declaration",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS,
        _METHOD_ROLE_NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, renderer: DirectBuildFindingRendererCandidate
    ) -> RefactorFinding:
        keyword_summary = ", ".join(renderer.keyword_names) or "no keywords"
        return self.build_finding(
            (
                f"`{renderer.class_name}.{renderer.method_name}` is a direct "
                f"`build_finding(...)` renderer with {renderer.positional_arg_count} "
                f"positional payloads and {keyword_summary}."
            ),
            (renderer.evidence,),
            scaffold=(
                "finding_renderer = CandidateFindingRenderer[Candidate](\n    summary=lambda candidate: ...,\n    evidence=lambda candidate: ...,\n)"
            ),
            codemod_patch=(
                f"# Move the `{renderer.method_name}` payload in `{renderer.class_name}` to a `CandidateFindingRenderer` classvar.\n# Let `CandidateFindingDetector._finding_for_candidate` run the renderer."
            ),
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=1,
                mapping_name=renderer.class_name,
                field_names=("summary", "evidence", *renderer.keyword_names),
            ),
        )


declare_candidate_rule_detector(
    DerivableDetectorIdCandidate,
    high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Detector id should derive from detector class name",
        "A detector class whose explicit `detector_id` is the snake_case projection of its class name is restating identity already available to the metaclass registry. `AutoRegisterMeta` should derive that key through the detector base.",
        "metaclass-derived detector registry key",
        "detector class repeats its own name as a manual registry key",
        _CLASS_LEVEL_REGISTRATION_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS,
        _CLASS_FAMILY_REGISTRY_POPULATION_REPEATED_METHOD_ROLES_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: f'`{candidate.class_name}` declares `detector_id = "{candidate.detector_id_value}"`, which is derivable from the class name.',
    scaffold=lambda candidate: "class IssueDetector(ABC, metaclass=AutoRegisterMeta):\n    __key_extractor__ = staticmethod(_detector_id_from_class_name)",
    codemod_patch=lambda candidate: f'# Delete `detector_id = "{candidate.detector_id_value}"` from `{candidate.class_name}` and let the metaclass derive it.',
    metrics=lambda candidate: RegistrationMetrics.from_class_names(
        registration_site_count=1,
        registry_name="IssueDetector",
        class_names=(candidate.class_name,),
    ),
    candidate_collector=_derivable_detector_id_candidates,
)


declare_candidate_rule_detector(
    DerivableCandidateCollectorCandidate,
    high_confidence_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Candidate collector should derive from detector class name",
        "A detector whose collector hook is the snake_case projection of its own class name is declaring a mechanical convention. The collector ABC can derive that hook at class creation and leave only non-standard collector aliases explicit.",
        "class-name-derived candidate collector hook",
        "detector class repeats its candidate collector naming convention",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS,
        _CLASS_FAMILY_REGISTRY_POPULATION_REPEATED_METHOD_ROLES_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: f"`{candidate.class_name}` declares `candidate_collector = {candidate.collector_name}`, which is derivable from the class name.",
    scaffold=lambda candidate: "class ModuleCollectorCandidateDetector(DerivedCandidateCollectorMixin, ...):\n    ...",
    codemod_patch=lambda candidate: f"# Delete `candidate_collector = {candidate.collector_name}` from `{candidate.class_name}` and let the collector ABC derive it.",
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=1,
        mapping_name=candidate.class_name,
        field_names=("candidate_collector",),
    ),
    candidate_collector=_derivable_candidate_collector_candidates,
)


declare_candidate_rule_detector(
    CanonicalFindingSpecBuilderCandidate,
    high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "FindingSpec coordinates should use one typed semantic builder",
        "Detector specs repeatedly enumerate the same semantic coordinate names: pattern, title, why, capability gap, relation context, and tag axes. A typed builder can make that product structure explicit once and leave each detector to provide only its coordinate values.",
        "typed FindingSpec builder with canonical semantic coordinate order",
        "detector repeats FindingSpec keyword schema locally",
        _AUTHORITATIVE_SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: f"`{candidate.class_name}` builds `{candidate.constructor_name}` by spelling {len(candidate.keyword_names)} FindingSpec coordinate keywords; use `{candidate.builder_name}`.",
    scaffold=lambda candidate: f"finding_spec = {candidate.builder_name}(\n    PatternId.EXAMPLE,\n    title,\n    why,\n    capability_gap,\n    relation_context,\n)",
    codemod_patch=lambda candidate: f"# Replace `{candidate.constructor_name}(pattern_id=..., title=..., ...)` with `{candidate.builder_name}(...)` and let the builder own coordinate names.",
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=1,
        mapping_name=candidate.class_name,
        field_names=candidate.keyword_names,
    ),
    candidate_collector=_canonical_finding_spec_builder_candidates,
)


declare_candidate_rule_detector(
    SortedTupleWrapperUseCandidate,
    high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "sorted_tuple wrapper should collapse to standard Python",
        "`sorted_tuple(...)` is only `tuple(sorted(...))` behind an extra project-local function. It hides ordinary Python collection semantics behind a nominal surface without adding type safety, registry authority, or an invariant.",
        "direct tuple(sorted(...)) expression instead of a project-local collection wrapper",
        "call site delegates ordinary sorted tuple construction to sorted_tuple",
        _AUTHORITATIVE_SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: f"`{candidate.qualname}` calls `sorted_tuple` with {candidate.argument_count} positional argument(s) and keywords {candidate.keyword_names}; use standard `tuple(sorted(...))` so the ordering operation remains AST-visible.",
    scaffold=lambda candidate: "value = tuple(sorted(items, key=key_function))",
    codemod_patch=lambda candidate: "# Replace `sorted_tuple(items, key=...)` with `tuple(sorted(items, key=...))` and delete the project-local wrapper once callers are gone.",
    metrics=lambda candidate: MappingMetrics(
        mapping_site_count=1,
        field_count=candidate.argument_count + len(candidate.keyword_names),
        mapping_name="sorted_tuple",
        field_names=("items", *candidate.keyword_names),
    ),
    candidate_collector=_sorted_tuple_wrapper_use_candidates,
)


declare_candidate_rule_detector(
    RuntimeProductRecordSchemaCandidate,
    high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Runtime product_record schemas should become AST-visible dataclasses",
        "`product_record` and `product_record_spec` create classes through runtime calls, which hides the class body, fields, inheritance, and docs from AST-level refactoring analysis. Field-only records should be explicit `@dataclass(frozen=True)` classes unless there is a stronger metaprogramming payoff proof.",
        "explicit frozen dataclass declaration visible to AST analysis",
        "runtime product_record schema hides an otherwise ordinary class declaration",
        _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: f"`{candidate.callee_name}` in `{candidate.context_qualname}` materializes product record(s) {candidate.declared_names or ('<dynamic>',)} over {candidate.line_count} line(s); spell these as explicit frozen dataclasses.",
    scaffold=lambda candidate: "from dataclasses import dataclass\n\n@dataclass(frozen=True)\nclass RecordName:\n    field_name: FieldType",
    codemod_patch=lambda candidate: "# Replace runtime `product_record` / `product_record_spec` materialization with explicit `@dataclass(frozen=True)` classes so NRA and other AST tooling can see the nominal structure.",
    metrics=lambda candidate: MappingMetrics(
        mapping_site_count=1,
        field_count=max(1, len(candidate.declared_names)),
        mapping_name=candidate.callee_name,
        field_names=candidate.declared_names or ("dynamic_product_record",),
    ),
    candidate_collector=_runtime_product_record_schema_candidates,
)


declare_candidate_rule_detector(
    SimplePropertyAliasClassCandidate,
    high_confidence_certified_spec(
        PatternId.LOCAL_VALUE_AUTHORITY,
        "Property alias class should use descriptor algebra",
        "A class whose only concrete behavior is returning `self.<field>` from properties is a structural alias shell. The alias relation is the semantic object; repeated property methods re-declare descriptor mechanics instead of naming that relation directly.",
        "typed alias-property descriptor derived from declared source and target names",
        "class repeats property method bodies for direct field projection",
        _SHARED_ALGORITHM_AUTHORITY_AUTHORITATIVE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: f"`{candidate.class_name}` defines {len(candidate.alias_pairs)} direct property alias(es) across {candidate.line_count} line(s): "
    + ", ".join((f"{target} -> {source}" for target, source in candidate.alias_pairs)),
    scaffold=lambda candidate: 'from nominal_refactor_advisor.descriptor_algebra import AliasProperty\n\nclass Shape:\n    target = AliasProperty[ValueType]("source")',
    codemod_patch=lambda candidate: "# Replace direct `@property return self.<source>` alias methods with `AliasProperty[...]` descriptors so alias projection is one typed descriptor algebra.",
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(candidate.alias_pairs),
        mapping_name=candidate.class_name,
        field_names=tuple(
            (f"{target}->{source}" for target, source in candidate.alias_pairs)
        ),
    ),
    candidate_collector=_simple_property_alias_class_candidates,
)


declare_candidate_rule_detector(
    SimplePropertyAliasMethodCandidate,
    high_confidence_certified_spec(
        PatternId.LOCAL_VALUE_AUTHORITY,
        "Direct property alias method should use descriptor algebra",
        "A property method whose body is exactly `return self.<field>` is a descriptor relation, even when the surrounding class owns other behavior. Keeping that relation as a method repeats alias mechanics and hides the source-target projection from class declarations.",
        "typed alias-property descriptor reused for direct field projection methods",
        "property method repeats direct self-field alias mechanics",
        _SHARED_ALGORITHM_AUTHORITY_AUTHORITATIVE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: f"`{candidate.class_name}.{candidate.method_name}` is a direct property alias to `{candidate.source_name}`.",
    scaffold=lambda candidate: f"""from nominal_refactor_advisor.descriptor_algebra import AliasProperty\n\n{candidate.method_name} = AliasProperty[{candidate.return_annotation or 'ValueType'}]("{candidate.source_name}")""",
    codemod_patch=lambda candidate: "# Replace the `@property return self.<source>` method with an `AliasProperty[...]` descriptor on the class body.",
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=1,
        mapping_name=f"{candidate.class_name}.{candidate.method_name}",
        field_names=(candidate.source_name,),
    ),
    candidate_collector=_simple_property_alias_method_candidates,
)


declare_candidate_rule_detector(
    SemanticTypeAliasCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Repeated structural type annotation should use a semantic alias",
        "A repeated nested structural annotation makes call signatures explain tuple/dict mechanics instead of domain roles. The semantic shape should be named once as a typed alias, then reused at fields, caches, and projector boundaries.",
        "named semantic type alias for repeated structural annotation shape",
        "same high-friction nested annotation appears at multiple semantic sites",
        _SHARED_ALGORITHM_AUTHORITY_AUTHORITATIVE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: (
        f"`{candidate.annotation_text}` appears in "
        f"{candidate.occurrence_count} annotations; name the domain shape once."
    ),
    evidence=lambda candidate: candidate.evidence_locations,
    scaffold=lambda candidate: (
        f"{candidate.suggested_alias_name} = {candidate.annotation_text}\n\n"
        "# Replace repeated annotation sites with the alias so signatures read "
        "in domain terms."
    ),
    codemod_patch=lambda candidate: (
        "# Introduce a module-level semantic type alias for the repeated "
        "annotation and replace each occurrence with that alias."
    ),
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=candidate.occurrence_count,
        mapping_name=candidate.suggested_alias_name,
        field_names=candidate.owner_symbols,
    ),
    candidate_collector=_semantic_type_alias_candidates,
    detector_priority=-10,
)


declare_candidate_rule_detector(
    SourceLocationEvidencePropertyCandidate,
    high_confidence_certified_spec(
        PatternId.LOCAL_VALUE_AUTHORITY,
        "SourceLocation evidence property should use descriptor algebra",
        "A property whose only behavior is constructing `SourceLocation` from three self attributes is a projection descriptor, not class-specific algorithm. The attribute triple should be data and the evidence-construction mechanics should live in one reusable descriptor.",
        "typed SourceLocation evidence descriptor parameterized by field names",
        "property method repeats evidence projection mechanics from self attributes",
        _SHARED_ALGORITHM_AUTHORITY_AUTHORITATIVE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: f"`{candidate.class_name}.{candidate.method_name}` constructs `SourceLocation` from `{candidate.file_attribute_name}`, `{candidate.line_attribute_name}`, `{candidate.symbol_attribute_name}`.",
    scaffold=lambda candidate: f'evidence = SourceLocationEvidenceProperty("{candidate.file_attribute_name}", "{candidate.line_attribute_name}", "{candidate.symbol_attribute_name}")',
    codemod_patch=lambda candidate: "# Replace the `@property` evidence method with `SourceLocationEvidenceProperty(...)`; keep only the irreducible attribute names as descriptor data.",
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=1,
        mapping_name=f"{candidate.class_name}.{candidate.method_name}",
        field_names=(
            candidate.file_attribute_name,
            candidate.line_attribute_name,
            candidate.symbol_attribute_name,
        ),
    ),
    candidate_collector=_source_location_evidence_property_candidates,
)


declare_candidate_rule_detector(
    ZippedSourceLocationEvidencePropertyCandidate,
    high_confidence_certified_spec(
        PatternId.LOCAL_VALUE_AUTHORITY,
        "Zipped SourceLocation evidence property should use descriptor algebra",
        "A property that zips parallel line and symbol fields only to construct `SourceLocation` tuples is a projection descriptor. The file/line/symbol field names are the semantic residue; the zip and construction mechanics should live in one reusable descriptor.",
        "typed zipped SourceLocation evidence descriptor parameterized by field names",
        "property method repeats zipped evidence projection mechanics from self attributes",
        _SHARED_ALGORITHM_AUTHORITY_AUTHORITATIVE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: f"`{candidate.class_name}.{candidate.method_name}` zips `{candidate.line_numbers_attribute_name}` with `{candidate.symbol_names_attribute_name}` to build SourceLocation evidence across {candidate.line_count} lines.",
    scaffold=lambda candidate: f'{candidate.method_name} = ZippedSourceLocationEvidenceProperty("{candidate.line_numbers_attribute_name}", "{candidate.symbol_names_attribute_name}", "{candidate.file_attribute_name}")',
    codemod_patch=lambda candidate: "# Replace the zipped evidence `@property` method with `ZippedSourceLocationEvidenceProperty(...)`; keep only the irreducible parallel field names as descriptor data.",
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=candidate.line_count,
        mapping_name=f"{candidate.class_name}.{candidate.method_name}",
        field_names=(
            candidate.file_attribute_name,
            candidate.line_numbers_attribute_name,
            candidate.symbol_names_attribute_name,
        ),
    ),
    candidate_collector=_zipped_source_location_evidence_property_candidates,
)


declare_candidate_rule_detector(
    PrivateHelperShadowCandidate,
    high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Private helper should reuse public package authority",
        "A private top-level helper whose name is only an underscore-prefixed version of a public helper in another package module is a local shadow of an existing authority. The private spelling should be removed or turned into an import alias so the shared helper owns the algorithm once.",
        "one public package helper authority reused by private call sites",
        "private helper repeats a public helper identity under an underscore-prefixed name",
        _AUTHORITATIVE_SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: f"`{candidate.private_name}` shadows public helper `{candidate.public_name}` from `{candidate.public_file_path}`.",
    evidence=lambda candidate: candidate.evidence,
    scaffold=lambda candidate: f"from package.module import {candidate.public_name} as {candidate.private_name}",
    codemod_patch=lambda candidate: f"# Delete local helper `{candidate.private_name}` and import `{candidate.public_name}` as the private compatibility name if call sites still use it.",
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=1,
        mapping_name=candidate.private_name,
        field_names=(candidate.public_name,),
        source_name=candidate.public_file_path,
    ),
    detector_base=CrossModuleCollectorCandidateDetector,
    detector_name="PrivateHelperShadowDetector",
    candidate_collector=_private_helper_shadow_candidates,
)


declare_candidate_rule_detector(
    OptionRecordQuotientCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Option record family should derive from one schema catalog",
        "Several small frozen option/config records in the same module are often projections of one closed format axis. Keeping every record as a hand-written class preserves type names, but repeats product mechanics and default surfaces that can be generated from a typed option schema catalog.",
        "typed option schema catalog that derives concrete option records",
        "field-only option/config record family repeats product-record mechanics across a closed format axis",
        _SHARED_ALGORITHM_AUTHORITY_AUTHORITATIVE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: f"{candidate.file_path} declares option record family {candidate.class_names} over fields {candidate.field_names}; derive the records from one typed option schema catalog.",
    scaffold=lambda candidate: "OPTION_SCHEMAS = (\n    OptionSchema('csv', CsvOptions, fields=(...)),\n    OptionSchema('json', JsonOptions, fields=(...)),\n)\n\n# Derive concrete frozen records and defaults from the schema catalog.",
    codemod_patch=lambda candidate: "# Keep the public option record names, but derive their field/default declarations from one schema catalog.\n# The only per-option residue should be semantic field/default differences.",
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(candidate.class_names),
        mapping_name="option_schema_catalog",
        field_names=candidate.field_names,
        identity_field_names=candidate.class_names,
    ),
    compression_certificate=_option_record_quotient_compression_certificate,
    detector_priority=-8,
    candidate_collector=_option_record_quotient_candidates,
)


declare_candidate_rule_detector(
    ClosedAxisConversionMatrixCandidate,
    high_confidence_certified_spec(
        PatternId.CLOSED_FAMILY_DISPATCH,
        "Conversion matrix should factor into source and target axes",
        "Functions named as pairwise conversions encode a product of two closed axes: source representation and target representation. The advisor should collapse the matrix into one dispatcher/table whose cases are derived from the axes instead of hand-maintaining one function per pair.",
        "closed source/target conversion axes with one derived dispatcher",
        "module declares many pairwise conversion functions whose names form a source-by-target matrix",
        _SHARED_ALGORITHM_AUTHORITY_AUTHORITATIVE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: f"{candidate.file_path} declares conversion matrix {candidate.function_names} over sources {candidate.source_axis_values} and targets {candidate.target_axis_values}; factor it into closed axes.",
    scaffold=lambda candidate: "class SourceMemory(Enum): ...\nclass TargetMemory(Enum): ...\n\nCONVERTERS = {\n    (SourceMemory.CPU, TargetMemory.GPU): convert_cpu_gpu,\n}\n\ndef convert(value, source, target):\n    return CONVERTERS[(source, target)](value)",
    codemod_patch=lambda candidate: "# Replace pairwise conversion function selection with one source/target axis table.\n# Keep specialized conversion bodies only as private table entries when they carry real backend semantics.",
    metrics=lambda candidate: DispatchCountMetrics(
        dispatch_site_count=len(candidate.function_names),
        dispatch_axis="source,target",
        literal_cases=(*candidate.source_axis_values, *candidate.target_axis_values),
    ),
    compression_certificate=_closed_axis_conversion_matrix_compression_certificate,
    detector_priority=-9,
    candidate_collector=_closed_axis_conversion_matrix_candidates,
)


def _bridge_axis_identifier(axis_expression: str) -> str:
    cleaned = "".join((ch if ch.isalnum() else "_" for ch in axis_expression))
    return cleaned.strip("_") or "representation"


def _bridge_axis_dispatch_scaffold(
    candidate: BridgeAxisDispatchFamilyCandidate,
) -> str:
    axis_identifier = _bridge_axis_identifier(candidate.axis_expression)
    return (
        "from abc import ABC, abstractmethod\n"
        "from metaclass_registry import AutoRegisterMeta\n\n"
        "class RepresentationBridge(ABC, metaclass=AutoRegisterMeta):\n"
        f'    __registry_key__ = "{axis_identifier}"\n'
        f"    {axis_identifier} = None\n\n"
        "    @classmethod\n"
        f"    def for_{axis_identifier}(cls, value):\n"
        "        return cls.__registry__[value]()\n\n"
        + "\n".join(
            (
                f"    @abstractmethod\n    def {operation_name}(self, value): ..."
                for operation_name in candidate.operation_names
            )
        )
    )


declare_candidate_rule_detector(
    BridgeAxisDispatchFamilyCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Repeated backend-axis dispatch should become a bridge authority",
        "Several operations redispatch the same closed representation axis. The ArrayBridge-style normal form is one ABC bridge family keyed by the representation, with shared lifecycle and operation hooks on the bridge implementations instead of every operation branching again.",
        "ABC bridge authority with one registered implementation per representation and operation hooks for the repeated actions",
        "multiple operations repeat the same backend/type/format dispatch axis",
        _SHARED_ALGORITHM_AUTHORITY_AUTHORITATIVE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: (
        f"Operations {candidate.function_names} all dispatch `{candidate.axis_expression}` "
        f"over cases {candidate.literal_cases}; factor the repeated axis into a bridge ABC "
        f"with hooks for {candidate.operation_names}."
    ),
    scaffold=_bridge_axis_dispatch_scaffold,
    codemod_patch=lambda candidate: (
        f"# Replace repeated `{candidate.axis_expression}` branches in {candidate.function_names} "
        "with one bridge lookup and operation hooks.\n"
        "# Keep backend-specific conversion/capability details on the bridge implementations; "
        "leave call sites responsible only for selecting the operation."
    ),
    metrics=lambda candidate: DispatchCountMetrics(
        dispatch_site_count=len(candidate.function_names),
        dispatch_axis=candidate.axis_expression,
        literal_cases=candidate.literal_cases,
    ),
    compression_certificate=lambda candidate: candidate.compression_certificate,
    detector_priority=-9,
    candidate_collector=_bridge_axis_dispatch_family_candidates,
)


declare_candidate_rule_detector(
    ArrayProtocolProbeBridgeCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Repeated array capability probes should become a bridge authority",
        "Several operations probe the same array protocol attributes. The bridge normal form is one nominal array bridge that owns capability discovery and exposes typed operation hooks, rather than every operation rediscovering shape/device/dtype semantics.",
        "array bridge ABC with capability properties and operation hooks",
        "multiple operations repeat the same array protocol capability probes",
        _SHARED_ALGORITHM_AUTHORITY_AUTHORITATIVE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: (
        f"Operations {candidate.function_names} repeat array capability probes "
        f"{candidate.attribute_names}; move capability discovery into an array bridge."
    ),
    scaffold=lambda candidate: (
        "class ArrayBridge(ABC):\n"
        + "\n".join(
            (
                f"    @property\n    @abstractmethod\n    def {attribute_name.strip('_')}(self): ..."
                for attribute_name in candidate.attribute_names
            )
        )
        + "\n\n    @abstractmethod\n    def normalize(self, value): ..."
    ),
    codemod_patch=lambda candidate: (
        f"# Replace repeated probes {candidate.attribute_names} in {candidate.function_names} "
        "with one array bridge selected at the boundary.\n"
        "# Keep protocol-specific dtype/device/shape logic behind bridge capability properties."
    ),
    metrics=lambda candidate: ProbeCountMetrics(probe_site_count=candidate.probe_count),
    compression_certificate=lambda candidate: candidate.compression_certificate,
    detector_priority=-9,
    candidate_collector=_array_protocol_probe_bridge_candidates,
)


declare_candidate_rule_detector(
    LifecycleStageSequenceCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Repeated lifecycle stage sequence should move into a template method",
        "Several functions execute the same ordered stage calls. That is a lifecycle skeleton: an ABC should own sequencing, while implementations provide hooks for the irreducible stages or payload residue.",
        "lifecycle ABC template method with stage hooks",
        "multiple operations repeat the same lifecycle stage sequence",
        _SHARED_ALGORITHM_AUTHORITY_AUTHORITATIVE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: (
        f"Functions {candidate.function_names} repeat lifecycle stages "
        f"{candidate.stage_names}; move sequencing into an ABC template method."
    ),
    scaffold=lambda candidate: (
        "class LifecycleTemplate(ABC):\n"
        "    def run(self, request):\n"
        + "\n".join(
            (
                f"        request = self.{stage_name}(request)"
                for stage_name in candidate.stage_names
            )
        )
        + "\n        return request"
    ),
    codemod_patch=lambda candidate: (
        f"# Move repeated stage order {candidate.stage_names} out of {candidate.function_names}.\n"
        "# Put the sequence in one ABC template method and leave only stage hooks or payload residue in implementations."
    ),
    metrics=lambda candidate: RepeatedMethodMetrics.from_duplicate_family(
        duplicate_site_count=len(candidate.function_names),
        statement_count=len(candidate.stage_names),
        class_count=1,
        method_symbols=candidate.function_names,
    ),
    compression_certificate=lambda candidate: candidate.compression_certificate,
    detector_priority=-9,
    candidate_collector=_lifecycle_stage_sequence_candidates,
)


declare_candidate_rule_detector(
    NodeVisitorStackBoilerplateCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Manual AST visitor scope stacks should be inherited",
        "Concrete `ast.NodeVisitor` classes that hand-declare multiple scope stacks and repeat push/pop transitions are reimplementing one traversal skeleton. The stack lifecycle belongs in a nominal ABC; concrete visitors should supply hooks for observation-specific work.",
        "one nominal visitor ABC owns stack lifecycle and concrete visitors provide hooks",
        "same visitor class declares multiple stack fields and push/pop transition hooks",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS,
        _NORMALIZED_AST_CLASS_FAMILY_METHOD_ROLE_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: f"`{candidate.qualname}` hand-declares visitor stacks {candidate.stack_names} across {candidate.transition_method_names}.",
    scaffold=lambda candidate: "class Visitor(ClassFunctionStackNodeVisitor):\n    def before_visit_function(self, node):\n        ...",
    codemod_patch=lambda candidate: "# Delete repeated stack lifecycle methods after moving initialization and `visit_ClassDef`/`visit_FunctionDef` push/pop transitions into a nominal ABC such as `ClassFunctionStackNodeVisitor`; keep only visitor-specific hooks and node handlers in the concrete class.",
    metrics=lambda candidate: RepeatedMethodMetrics.from_duplicate_family(
        duplicate_site_count=len(candidate.transition_method_names),
        statement_count=candidate.line_count,
        class_count=1,
        method_symbols=tuple(
            (
                f"{candidate.qualname}.{method_name}"
                for method_name in candidate.transition_method_names
            )
        ),
    ),
    candidate_collector=_node_visitor_stack_boilerplate_candidates,
)


declare_candidate_rule_detector(
    DuplicateVisitorMethodBodyCandidate,
    high_confidence_certified_spec(
        PatternId.ABC_TEMPLATE_METHOD,
        "Duplicate AST visitor hooks should share one hook implementation",
        "Sibling `visit_*` methods with exactly the same normalized body encode one visitor transition more than once. The shared body should be one hook or an explicit method alias, leaving the node-type distinction in dispatch metadata.",
        "one visitor hook implementation reused by equivalent node dispatch entries",
        "same normalized visitor hook body is repeated on sibling visit methods",
        _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS,
        _NORMALIZED_AST_CLASS_FAMILY_METHOD_ROLE_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: f"`{candidate.class_name}` repeats the same visitor body across {', '.join(candidate.method_names)}.",
    scaffold=lambda candidate: f"{candidate.method_names[0]}(...)\n{candidate.method_names[1]} = {candidate.method_names[0]}",
    codemod_patch=lambda candidate: "# Replace duplicate sibling `visit_*` method bodies with one shared implementation or explicit aliases for equivalent visitor dispatch entries.",
    metrics=lambda candidate: RepeatedMethodMetrics.from_duplicate_family(
        duplicate_site_count=len(candidate.method_names),
        statement_count=candidate.statement_count,
        class_count=1,
        method_symbols=tuple(
            (
                f"{candidate.class_name}.{method_name}"
                for method_name in candidate.method_names
            )
        ),
    ),
    candidate_collector=_duplicate_visitor_method_body_candidates,
)


declare_candidate_rule_detector(
    EnumMetadataTableCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Enum metadata table should be carried by enum members",
        "An enum whose properties only index a module-level table by `self` splits member identity from member metadata. The metadata should move into enum construction so each member carries its own typed semantic record.",
        "enum member construction owns the member metadata",
        "enum properties read a parallel metadata table keyed by the same enum family",
        _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
        _EXPORT_MAPPING_NORMALIZED_AST_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: f"`{candidate.class_name}` reads {candidate.property_names} from `{candidate.table_name}` across {candidate.case_count} enum cases.",
    scaffold=lambda candidate: "class MetadataEnum(StrEnum):\n    def __new__(cls, value: str, label: str):\n        obj = str.__new__(cls, value)\n        obj._value_ = value\n        obj.label = label\n        return obj",
    codemod_patch=lambda candidate: f"# Move `{candidate.table_name}` values into `{candidate.class_name}` member tuples and delete the table-backed property lookups.",
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=candidate.case_count,
        mapping_name=candidate.table_name,
        field_names=candidate.property_names,
        source_name=candidate.class_name,
    ),
    candidate_collector=_enum_metadata_table_candidates,
)


class SemanticTagTupleBoilerplateDetector(
    ModuleCollectorCandidateDetector[SemanticTagTupleBoilerplateCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Semantic tag tuple literal should become a named authority",
        "Capability and observation tag tuples are semantic classifications, not local control flow. A named constant should carry that classification so detector specs reference one authority instead of re-declaring enum tuples inline.",
        "named semantic tag tuple authority reused across detector specs",
        "FindingSpec carries an inline semantic tag tuple literal",
        _AUTHORITATIVE_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, tag_candidate: SemanticTagTupleBoilerplateCandidate
    ) -> RefactorFinding:
        if tag_candidate.source_kind == "derived_constant":
            sample_names = ", ".join(tag_candidate.tag_names[:4])
            suffix = (
                f", and {len(tag_candidate.tag_names) - 4} more"
                if len(tag_candidate.tag_names) > 4
                else ""
            )
            return self.build_finding(
                (
                    f"{len(tag_candidate.tag_names)} {tag_candidate.keyword_name} tag constants "
                    f"restate enum tuples already encoded by their names: {sample_names}{suffix}."
                ),
                tag_candidate.evidence,
                scaffold="globals().update({name: _semantic_tag_tuple_from_constant_name(name) for name in names})",
                codemod_patch="# Delete the manual constants; add one typed derivation helper for capability/observation tag constants.",
                metrics=MappingMetrics.from_field_names(
                    mapping_site_count=len(tag_candidate.tag_names),
                    mapping_name=tag_candidate.constant_name,
                    field_names=tag_candidate.tag_names,
                ),
            )
        return self.build_finding(
            (
                f"`{tag_candidate.keyword_name}` repeats tag tuple {tag_candidate.tag_names}; "
                f"lift it to `{tag_candidate.constant_name}`."
            ),
            tag_candidate.evidence,
            scaffold=(
                f"{tag_candidate.constant_name} = (\n    ...\n)\n\nfinding_spec = HighConfidenceFindingSpec({tag_candidate.keyword_name}={tag_candidate.constant_name})"
            ),
            codemod_patch=(
                f"# Add `{tag_candidate.constant_name}` for the repeated tag tuple.\n"
                f"# Replace each repeated `{tag_candidate.keyword_name}=(...)` value with the named constant."
            ),
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(tag_candidate.evidence),
                mapping_name=tag_candidate.constant_name,
                field_names=tag_candidate.tag_names,
            ),
        )


class DerivedMetricCountBoilerplateDetector(
    ModuleCollectorCandidateDetector[DerivedMetricCountBoilerplateCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Metric counts should be derived from metric collections",
        "A metrics object that receives both `*_count=len(values)` and `values=values` is carrying the same fact twice. The count is a deterministic projection of the collection and should be derived by the typed metrics constructor.",
        "typed metrics constructors that derive count fields from authoritative collections",
        "metrics call passes a count keyword computed from the collection keyword in the same call",
        _AUTHORITATIVE_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS,
    )

    def _finding_for_candidate(
        self, metric_candidate: DerivedMetricCountBoilerplateCandidate
    ) -> RefactorFinding:
        derived_summary = ", ".join(
            (
                f"{count_name}=len({collection_name})"
                for count_name, collection_name in zip(
                    metric_candidate.count_keyword_names,
                    metric_candidate.collection_keyword_names,
                    strict=True,
                )
            )
        )
        return self.build_finding(
            (
                f"`{metric_candidate.metric_class_name}` repeats derived count fields "
                f"{derived_summary}; use `{metric_candidate.recommended_constructor_name}`."
            ),
            (metric_candidate.evidence,),
            scaffold=(
                f"{metric_candidate.metric_class_name}.{metric_candidate.recommended_constructor_name}(\n    ...\n)"
            ),
            codemod_patch=(
                f"# Replace `{metric_candidate.metric_class_name}(...)` with "
                f"`{metric_candidate.metric_class_name}.{metric_candidate.recommended_constructor_name}(...)`.\n"
                f"# Delete derived count keywords: {', '.join(metric_candidate.count_keyword_names)}."
            ),
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=len(metric_candidate.count_keyword_names),
                mapping_name=metric_candidate.metric_class_name,
                field_names=metric_candidate.collection_keyword_names,
            ),
        )


declare_candidate_rule_detector(
    DataclassNamespaceCliMirrorCandidate,
    high_confidence_certified_spec(
        PatternId.AUTHORITATIVE_SCHEMA,
        "Dataclass config surfaces should derive namespace and CLI adapters",
        "A dataclass already owns its field names and defaults. Re-enumerating those fields in a namespace constructor and an argparse specification table creates parallel configuration surfaces that can drift from the typed record.",
        "one dataclass field authority that derives namespace construction and CLI argument rows",
        "dataclass fields are mirrored manually in both from-namespace construction and CLI argument specs",
        _AUTHORITATIVE_NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS,
        _DATAFLOW_ROOT_NORMALIZED_AST_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS,
    ),
    summary=lambda candidate: f"`{candidate.class_name}` mirrors {len(candidate.field_names)} namespace fields and {len(candidate.cli_field_names)} CLI fields through `{candidate.argument_spec_name}` instead of deriving adapters from the dataclass.",
    evidence=lambda candidate: (
        SourceLocation(candidate.file_path, candidate.line, candidate.class_name),
        SourceLocation(
            candidate.file_path,
            candidate.from_namespace_line,
            f"{candidate.class_name}.from_namespace",
        ),
        SourceLocation(
            candidate.argument_spec_file_path,
            candidate.argument_spec_line,
            candidate.argument_spec_name,
        ),
    ),
    scaffold=lambda candidate: "for field in fields(ConfigRecord):\n    value = namespace_values.get(field.name, field.default)\n    ...\n\nCLI_SPECS = tuple(spec_from_field(field) for field in fields(ConfigRecord) if field.name in HELP)",
    codemod_patch=lambda candidate: f"# Derive `{candidate.class_name}.from_namespace()` and `{candidate.argument_spec_name}` from dataclass fields/defaults.\n# Keep per-option help text as the only CLI-specific residue.",
    metrics=lambda candidate: MappingMetrics.from_field_names(
        mapping_site_count=len(candidate.field_names) + len(candidate.cli_field_names),
        mapping_name=candidate.class_name,
        field_names=candidate.field_names,
        source_name=candidate.argument_spec_name,
    ),
    detector_base=CrossModuleCollectorCandidateDetector,
    candidate_collector=_dataclass_namespace_cli_mirror_candidates,
)


class NestedBuilderShellDetector(
    ConfiguredModuleCollectorCandidateDetector[NestedBuilderShellCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.AUTHORITATIVE_CONTEXT,
        "Nested builder shell should collapse into one authoritative request boundary",
        "A builder forwards a substantial semantic parameter family unchanged into a subordinate nominal builder and only adds a small residue locally. The docs treat that as split request authority: one layer should own the forwarded family instead of rebuilding it inside another shell.",
        "single authoritative request/context builder boundary",
        "one builder nests a forwarded subordinate request builder inside a second nominal shell",
        _AUTHORITATIVE_PROVENANCE_UNIT_RATE_COHERENCE_CAPABILITY_TAGS,
    )

    def _finding_for_candidate(
        self, shell_candidate: NestedBuilderShellCandidate
    ) -> RefactorFinding:
        forwarded = ", ".join(shell_candidate.forwarded_parameter_names)
        residue_fields = ", ".join(shell_candidate.residue_field_names)
        residue_sources = ", ".join(shell_candidate.residue_source_names)
        return self.build_finding(
            (
                f"`{shell_candidate.qualname}` forwards `{forwarded}` into "
                f"`{shell_candidate.nested_callee_name}` under `{shell_candidate.nested_field_name}` "
                f"while separately deriving `{residue_fields}` from `{residue_sources}`."
            ),
            (shell_candidate.evidence,),
            scaffold=(
                "@dataclass(frozen=True)\nclass OuterRequest:\n    child_request: ChildRequest\n\n    @classmethod\n    def from_source(cls, source, *, child_request: ChildRequest):\n        return cls(child_request=child_request, ...)\n"
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


declare_candidate_rule_detector(
    ManualFiberTagCandidate,
    high_confidence_spec(
        PatternId.NOMINAL_BOUNDARY,
        "Manual fiber tag should become nominal family",
        "A string-valued instance tag is manually selecting behavior while the same instance still carries fields from several incompatible fibers. That leaves the family above the zero-incoherence threshold and admits disagreement states the host type system could rule out.",
        "host-native nominal fiber decomposition with one subclass per behavior fiber",
        "manual instance tag drives behavior while irrelevant coordinates remain constructible on every fiber",
        _NOMINAL_IDENTITY_PROVENANCE_FAIL_LOUD_CONTRACTS_CAPABILITY_TAGS,
    ),
    summary=lambda fiber_candidate: f"`{fiber_candidate.class_name}` branches on manual fiber tag `self.{fiber_candidate.tag_name}` across {fiber_candidate.case_names} while still carrying cross-fiber fields {fiber_candidate.assigned_field_names}.",
    evidence=lambda fiber_candidate: (
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
    scaffold=lambda fiber_candidate: _manual_fiber_tag_scaffold(fiber_candidate),
    codemod_patch=lambda fiber_candidate: _manual_fiber_tag_patch(fiber_candidate),
    metrics=lambda fiber_candidate: DispatchCountMetrics.from_literal_family(
        dispatch_axis=f"self.{fiber_candidate.tag_name}",
        literal_cases=fiber_candidate.case_names,
    ),
    candidate_collector=_manual_fiber_tag_candidates,
)


declare_candidate_rule_detector(
    DescriptorDerivedViewCandidate,
    high_confidence_spec(
        PatternId.DESCRIPTOR_DERIVED_VIEW,
        "Derived views stored independently of their source",
        "Several stored fields are derived from one authoritative source field, but mutators resynchronize them manually and incompletely. That raises the degree of freedom above one and makes view disagreement reachable.",
        "descriptor- or property-mediated derived views rooted in one authoritative source",
        "stored derived views must be manually kept coherent with a single source field",
        _AUTHORITATIVE_UNIT_RATE_COHERENCE_PROVENANCE_CAPABILITY_TAGS,
    ),
    summary=lambda view_candidate: f"`{view_candidate.class_name}` stores derived views {view_candidate.derived_field_names} from `{view_candidate.source_attr}`, but `{view_candidate.mutator_name}` only updates {view_candidate.updated_field_names}.",
    evidence=lambda view_candidate: (
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
    scaffold=lambda view_candidate: _descriptor_derived_view_scaffold(view_candidate),
    codemod_patch=lambda view_candidate: _descriptor_derived_view_patch(view_candidate),
    candidate_collector=_descriptor_derived_view_candidates,
)


class DeferredClassRegistrationDetector(
    ModuleCollectorCandidateDetector[ManualRegistryCandidate]
):
    candidate_collector = _manual_registry_candidates
    finding_spec = high_confidence_spec(
        PatternId.AUTO_REGISTER_META,
        "Class registration is decoupled from class existence",
        "Manual decorator- or helper-based registration leaves a reachable state where a class exists but the registry has not been updated. The host already provides zero-delay registration via `metaclass-registry` or another class-time hook.",
        "zero-delay metaclass-registry class registration with collision checks and runtime provenance",
        "class registration is performed as a separate auxiliary step rather than at class creation time",
        _CLASS_LEVEL_REGISTRATION_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS,
    )

    def _finding_for_candidate(
        self, registry_candidate: ManualRegistryCandidate
    ) -> RefactorFinding:
        evidence = [
            SourceLocation(
                registry_candidate.file_path,
                registry_candidate.line,
                registry_candidate.decorator_name,
            )
        ]
        evidence.extend(
            (
                SourceLocation(
                    registry_candidate.file_path, registry_candidate.line, class_name
                )
                for class_name in registry_candidate.class_names[:5]
            )
        )
        return self.build_finding(
            f"Registry `{registry_candidate.registry_name}` is updated through manual decorator `{registry_candidate.decorator_name}` for classes {registry_candidate.class_names}, leaving registration structurally decoupled from class creation.",
            tuple(evidence),
            scaffold=_manual_registry_scaffold(registry_candidate),
            codemod_patch=_manual_registry_patch(registry_candidate),
            metrics=RegistrationMetrics(
                registration_site_count=len(registry_candidate.class_names),
                registry_name=registry_candidate.registry_name,
            ),
        )


class StructuralConfusabilityDetector(
    ModuleCollectorCandidateDetector[StructuralConfusabilityCandidate]
):
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_INTERFACE_WITNESS,
        "Consumer observes a confusable duck-typed family",
        "A consumer only observes a partial structural view, and several unrelated classes are confusable under that view. Without a nominal witness, the distortion floor stays above zero and the family boundary remains implicit.",
        "ABC-backed nominal witness for a structurally confusable implementation family",
        "consumer depends on a partial structural view shared by several unrelated classes",
        _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_PROVENANCE_CAPABILITY_TAGS,
    )

    def _finding_for_candidate(
        self, confusability_candidate: StructuralConfusabilityCandidate
    ) -> RefactorFinding:
        evidence = (
            SourceLocation(
                confusability_candidate.file_path,
                confusability_candidate.line,
                confusability_candidate.function_name,
            ),
        )
        return self.build_finding(
            f"`{confusability_candidate.function_name}` observes `{confusability_candidate.parameter_name}` only through methods {confusability_candidate.observed_method_names}, but classes {confusability_candidate.class_names} are confusable under that view.",
            evidence,
            scaffold=_structural_confusability_scaffold(confusability_candidate),
            codemod_patch=_structural_confusability_patch(confusability_candidate),
        )


class SemanticWitnessFamilyDetector(
    ModuleCollectorCandidateDetector[WitnessCarrierFamilyCandidate]
):
    candidate_collector = _witness_carrier_family_candidates
    finding_spec = high_confidence_spec(
        PatternId.NOMINAL_WITNESS_CARRIER,
        "Semantic carrier family should share one nominal base",
        "Several frozen dataclass carriers repeat the same location and naming roles under different field names. That leaves one semantic family structurally expanded instead of giving it one nominal carrier root.",
        "one authoritative nominal base for a semantic metadata carrier family",
        "same carrier family repeats a renamed semantic-role spine across sibling frozen dataclasses",
        _NOMINAL_IDENTITY_PROVENANCE_AUTHORITATIVE_CAPABILITY_TAGS,
    )

    def _finding_for_candidate(
        self, witness_candidate: WitnessCarrierFamilyCandidate
    ) -> RefactorFinding:
        evidence = tuple(
            (
                SourceLocation(witness_candidate.file_path, line, class_name)
                for class_name, line in zip(
                    witness_candidate.class_names,
                    witness_candidate.line_numbers,
                    strict=True,
                )
            )
        )
        return self.build_finding(
            f"Frozen carrier classes {', '.join(witness_candidate.class_names)} repeat semantic roles {witness_candidate.shared_role_names} under renamed fields and should inherit one nominal base carrier.",
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
