"""Systemic detector implementations.

This module holds the earlier detector families that focus on orchestration,
axis authority, registration, and other repo-wide architectural smells.
"""
from __future__ import annotations
from ..record_algebra import product_record
from ._base import *
from ._helpers import *
def _imported_name_aliases(module: ast.Module, *, module_names: frozenset[str], imported_name: str) -> frozenset[str]:
    aliases: set[str] = {imported_name}
    for node in module.body:
        if not isinstance(node, ast.ImportFrom) or node.module not in module_names: continue
        for alias in node.names:
            if alias.name == imported_name: aliases.add(alias.asname or alias.name)
    return frozenset(aliases)
def _module_aliases(module: ast.Module, module_names: frozenset[str]) -> frozenset[str]:
    aliases: set[str] = set()
    for node in module.body:
        if not isinstance(node, ast.Import): continue
        for alias in node.names:
            if alias.name in module_names: aliases.add(alias.asname or alias.name.split('.', maxsplit=1)[0])
    return frozenset(aliases)
def _is_imported_name(expr: ast.AST, aliases: frozenset[str]) -> bool: return isinstance(expr, ast.Name) and expr.id in aliases
def _is_qualified_name(expr: ast.AST, *, module_aliases: frozenset[str], attr_name: str) -> bool: return isinstance(expr, ast.Attribute) and expr.attr == attr_name and isinstance(expr.value, ast.Name) and (expr.value.id in module_aliases)
_SINGLE_TEMPLATE_CALL_METRICS = OrchestrationMetrics(function_line_count=0, branch_site_count=0, call_site_count=1, parameter_count=1, callee_family_count=1)
class TypingProtocolContractDetector(IssueDetector):
    detector_priority = -20
    finding_spec = high_confidence_spec(PatternId.ABC_TEMPLATE_METHOD, 'Structural Protocol contract should be a nominal ABC', "`typing.Protocol` is structural: it lets values claim membership by shape rather than by a declared nominal contract. The advisor's nominal architecture rules should route those interfaces through ABCs, explicit subclassing, or ABC virtual registration instead.", 'nominal runtime contract instead of structural shape membership', 'class declares interface identity through structural typing', _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_VIRTUAL_MEMBERSHIP_CAPABILITY_TAGS, _CLASS_FAMILY_RUNTIME_MEMBERSHIP_OBSERVATION_TAGS)
    def _collect_findings(self, modules: list[ParsedModule], config: DetectorConfig) -> list[RefactorFinding]:
        findings: list[RefactorFinding] = []
        typing_modules = frozenset({"typing", "typing_extensions"})
        for module in modules:
            protocol_aliases = _imported_name_aliases(module.module, module_names=typing_modules, imported_name='Protocol')
            runtime_checkable_aliases = _imported_name_aliases(module.module, module_names=typing_modules, imported_name='runtime_checkable')
            typing_aliases = _module_aliases(module.module, typing_modules)
            evidence: list[SourceLocation] = []
            protocol_class_names: list[str] = []
            for node in ast.walk(module.module):
                if not isinstance(node, ast.ClassDef): continue
                inherits_protocol = any((_is_imported_name(base, protocol_aliases) or _is_qualified_name(base, module_aliases=typing_aliases, attr_name='Protocol') for base in node.bases))
                runtime_checkable = any((_is_imported_name(decorator, runtime_checkable_aliases) or _is_qualified_name(decorator, module_aliases=typing_aliases, attr_name='runtime_checkable') for decorator in node.decorator_list))
                if inherits_protocol:
                    protocol_class_names.append(node.name)
                    evidence.append(SourceLocation(str(module.path), node.lineno, node.name))
                elif runtime_checkable:
                    evidence.append(SourceLocation(str(module.path), node.lineno, f'{node.name}.runtime_checkable'))
            if not evidence: continue
            class_summary = ", ".join(protocol_class_names) or "runtime-checkable class"
            findings.append(
                self.build_finding(
                    (
                        f"{module.path} declares structural typing interfaces "
                        f"({class_summary}); replace them with ABC-backed nominal contracts."
                    ),
                    tuple(evidence[:8]),
                    scaffold=(
                        'from abc import ABC, abstractmethod\n\nclass ContractName(ABC):\n    @abstractmethod\n    def required_method(self, request): ...\n\n# Use direct subclassing for owned implementations, or `ContractName.register(ExternalType)` for explicit virtual membership.'
                    ),
                )
            )
        return findings
class RepeatedPrivateMethodDetector(FiberCollectedShapeIssueDetector):
    detector_id = "repeated_private_methods"
    observation_kind = ObservationKind.METHOD_SHAPE
    finding_spec = high_confidence_certified_spec(PatternId.ABC_TEMPLATE_METHOD, 'Repeated non-orthogonal method skeleton across classes', 'Shared orchestration logic is duplicated across a behavior family. The docs say this shared non-orthogonal logic should move into an ABC with a concrete template method, leaving only orthogonal hooks in subclasses.', 'single authoritative algorithm for a nominal behavior family', 'same method role across sibling classes', _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS, _NORMALIZED_AST_CLASS_FAMILY_METHOD_ROLE_OBSERVATION_TAGS)
    def _module_shapes(self, module: ParsedModule) -> tuple[object, ...]: return tuple(_collect_typed_family_items(module, MethodShapeFamily, MethodShape))
    def _include_shape(self, shape: object, config: DetectorConfig) -> bool: method = _as_method_shape(shape); return bool(method.class_name and method.statement_count >= config.min_duplicate_statements)
    def _group_key(self, shape: object) -> object: method = _as_method_shape(shape); return (method.is_private, method.param_count, method.fingerprint)
    def _finding_from_group(self, shapes: tuple[object, ...], config: DetectorConfig) -> RefactorFinding | None:
        methods = sorted_tuple((_as_method_shape(shape) for shape in shapes), key=lambda item: (item.file_path, item.lineno))
        class_names = {method.class_name for method in methods}
        if len(methods) < 2 or len(class_names) < 2: return None
        evidence = tuple((SourceLocation(method.file_path, method.lineno, method.symbol) for method in methods[:6]))
        relation = (
            "same private helper role across sibling classes"
            if methods[0].is_private
            else "same method role across sibling classes"
        )
        return self.build_finding(f'{len(methods)} methods across {len(class_names)} classes share the same normalized AST shape.', evidence, relation_context=relation, scaffold=_abc_scaffold_for_methods(methods), codemod_patch=_abc_patch_for_methods(methods), metrics=RepeatedMethodMetrics.from_duplicate_family(duplicate_site_count=len(methods), statement_count=methods[0].statement_count, class_count=len(class_names), method_symbols=tuple((method.symbol for method in methods)), shared_statement_texts=methods[0].statement_texts))
class InheritanceHierarchyCandidateDetector(IssueDetector):
    finding_spec = high_confidence_spec(PatternId.ABC_TEMPLATE_METHOD, 'Classes cluster into an ABC hierarchy candidate', 'The same set of classes repeats multiple non-orthogonal method skeletons. The docs say this is a strong signal that the family should be factored into an ABC with one concrete template method layer; orthogonal reusable concerns can then live in mixins so MRO preserves declared precedence.', 'single authoritative inheritance hierarchy for a duplicated behavior family', 'same class set repeats several method roles across the same family boundary', _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS, _REPEATED_METHOD_ROLES_CLASS_FAMILY_NORMALIZED_AST_OBSERVATION_TAGS)
    def _collect_findings(self, modules: list[ParsedModule], config: DetectorConfig) -> list[RefactorFinding]:
        repeated_methods = tuple((method for module in modules for method in _collect_typed_family_items(module, MethodShapeFamily, MethodShape) if method.class_name and method.statement_count >= config.min_duplicate_statements))
        graph = ObservationGraph(tuple((method.structural_observation for method in repeated_methods)))
        lookup = _carrier_lookup(tuple(repeated_methods))
        findings: list[RefactorFinding] = []
        for cohort in graph.coherence_cohorts_for(ObservationKind.METHOD_SHAPE, StructuralExecutionLevel.FUNCTION_BODY, minimum_witnesses=2, minimum_fibers=1):
            groups = [
                tuple((_as_method_shape(item) for item in _materialize_observations(fiber.observations, lookup)))
                for fiber in cohort.fibers
            ]
            if not groups: continue
            class_names = frozenset(cohort.nominal_witnesses)
            method_count_by_class: dict[str, int] = defaultdict(int)
            for methods in groups:
                for method in methods:
                    if method.class_name is not None: method_count_by_class[method.class_name] += 1
            supports_family = (
                len(groups) >= 2
                or sum(1 for count in method_count_by_class.values() if count >= 2) >= 2
            )
            if not supports_family: continue
            evidence = [
                SourceLocation(method.file_path, method.lineno, method.symbol)
                for methods in groups
                for method in methods
            ]
            findings.append(self.build_finding(f"Classes {', '.join(sorted(class_names))} share {len(groups)} repeated method-shape groups and repeated method roles that likely want one ABC family.", tuple(evidence[:8]), scaffold=_abc_family_scaffold(class_names, groups), codemod_patch=_abc_family_patch(class_names, groups), metrics=HierarchyCandidateMetrics(duplicate_group_count=len(groups), class_count=len(class_names))))
        return findings
class OrchestrationHubDetector(CandidateFindingDetector[FunctionProfile]):
    finding_spec = high_confidence_spec(PatternId.STAGED_ORCHESTRATION, 'Oversized orchestration hub', 'One function is owning too many control branches, helper calls, and phase transitions at once. The architecture wants explicit staged boundaries so the orchestration surface remains nominal and legible.', 'explicit staged orchestration boundaries with named phase contracts', 'one owner centralizes many operational phases and helper families', _SHARED_ALGORITHM_AUTHORITY_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS)
    def _candidate_items(self, module: ParsedModule, config: DetectorConfig) -> Sequence[object]: return tuple((profile for profile in _function_profiles(module) if profile.line_count >= config.min_orchestration_function_lines and profile.branch_count >= config.min_orchestration_branches and (profile.call_count >= config.min_orchestration_calls)))
    finding_renderer = CandidateFindingRenderer[FunctionProfile](summary=lambda profile: f'`{profile.qualname}` concentrates {profile.line_count} lines, {profile.branch_count} branches, and {profile.call_count} calls across {profile.callee_family_count} callee families in one owner.', evidence=lambda profile: (profile.evidence,), scaffold=lambda profile: _orchestration_stage_scaffold(profile), codemod_patch=lambda profile: _orchestration_stage_patch(profile), metrics=lambda profile: OrchestrationMetrics(function_line_count=profile.line_count, branch_site_count=profile.branch_count, call_site_count=profile.call_count, parameter_count=len(profile.parameter_names), callee_family_count=profile.callee_family_count))
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
    def role_names(self) -> tuple[str, ...]: return tuple((role for role, _ in self.role_method_counts))
    @property
    def evidence_locations(self) -> tuple[SourceLocation, ...]: return tuple((SourceLocation(self.file_path, line, f'{self.class_name}.{method_name}') for role, line, method_name in self.role_representatives))
def _method_role_token(method_name: str) -> str:
    stripped = method_name.strip("_")
    if not stripped: return ''
    return stripped.split("_", maxsplit=1)[0]
def _class_role_quotient_candidates(module: ParsedModule) -> tuple[ClassRoleQuotientCandidate, ...]:
    candidates: list[ClassRoleQuotientCandidate] = []
    for class_node in (
        node for node in ast.walk(module.module) if isinstance(node, ast.ClassDef)
    ):
        methods = tuple((statement for statement in class_node.body if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and (not (statement.name.startswith('__') and statement.name.endswith('__')))))
        if not methods: continue
        role_groups: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = (
            defaultdict(list)
        )
        method_role: dict[str, str] = {}
        for method in methods:
            role = _method_role_token(method.name)
            if not role: continue
            role_groups[role].append(method)
            method_role[method.name] = role
        nontrivial_roles = {
            role: role_methods
            for role, role_methods in role_groups.items()
            if len(role_methods) >= 2
        }
        nontrivial_method_count = sum((len(role_methods) for role_methods in nontrivial_roles.values()))
        private_method_count = sum((1 for method in methods if method.name.startswith('_')))
        public_method_count = len(methods) - private_method_count
        if len(nontrivial_roles) < 3: continue
        if private_method_count <= public_method_count: continue
        if nontrivial_method_count * 2 < len(methods): continue
        self_state_attributes: set[str] = set()
        self_call_count = 0
        cross_role_self_call_count = 0
        method_names = {method.name for method in methods}
        for method in methods:
            caller_role = method_role.get(method.name, "")
            for child in ast.walk(method):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id == "self"
                    and child.func.attr in method_names
                ):
                    self_call_count += 1
                    callee_role = method_role.get(child.func.attr, "")
                    if callee_role and caller_role and (callee_role != caller_role): cross_role_self_call_count += 1
                elif (
                    isinstance(child, ast.Attribute)
                    and isinstance(child.value, ast.Name)
                    and child.value.id == "self"
                    and child.attr not in method_names
                ):
                    self_state_attributes.add(child.attr)
        role_method_counts = sorted_tuple(((role, len(role_methods)) for role, role_methods in nontrivial_roles.items()), key=lambda item: (-item[1], item[0]))
        representatives: list[tuple[str, int, str]] = []
        for role, _ in role_method_counts:
            role_methods = sorted(nontrivial_roles[role], key=lambda method: method.lineno)
            representative = role_methods[0]
            representatives.append((role, representative.lineno, representative.name))
        candidates.append(ClassRoleQuotientCandidate(file_path=str(module.path), line=class_node.lineno, class_name=class_node.name, method_count=len(methods), private_method_count=private_method_count, public_method_count=public_method_count, role_method_counts=role_method_counts, role_representatives=tuple(representatives[:8]), self_state_attribute_count=len(self_state_attributes), self_call_count=self_call_count, cross_role_self_call_count=cross_role_self_call_count))
    return sorted_tuple(candidates, key=lambda candidate: (candidate.file_path, candidate.line, candidate.class_name))
class ClassRoleQuotientDetector(ModuleCollectorCandidateDetector[ClassRoleQuotientCandidate]):
    finding_spec = high_confidence_spec(PatternId.STAGED_ORCHESTRATION, 'Class method-role quotient should become composed subsystems', 'The class method namespace factors into several nontrivial role-equivalence classes, while private methods dominate the public facade. That quotient is a formal signal that one nominal object is carrying a product of subsystem algebras instead of composing role-owned services.', 'composed subsystem authorities derived from the class method-role quotient', 'one class contains several nontrivial method-role equivalence classes behind a smaller public facade', _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS, _METHOD_ROLE_DATAFLOW_ROOT_PARTIAL_VIEW_OBSERVATION_TAGS)
    def _finding_for_candidate(self, role_candidate: ClassRoleQuotientCandidate) -> RefactorFinding:
        role_summary = ', '.join((f'{role}:{count}' for role, count in role_candidate.role_method_counts[:8]))
        return self.build_finding(
            (
                f"Class `{role_candidate.class_name}` has {role_candidate.method_count} methods "
                f"whose nontrivial method-role quotient is {role_summary}; "
                f"{role_candidate.private_method_count} private methods sit behind "
                f"{role_candidate.public_method_count} public methods."
            ),
            role_candidate.evidence_locations,
            scaffold=(
                '@dataclass(frozen=True)\nclass BuilderContext:\n    ...\n\nclass RoleSubsystem:\n    def __init__(self, context: BuilderContext): ...\n\nclass Facade:\n    def __init__(self, context: BuilderContext):\n        self.role = RoleSubsystem(context)\n\n# Factor the role quotient into composed subsystem authorities. Leave the original class as a public facade and sequencing boundary.'
            ),
            codemod_patch=(
                '# Partition the class by the method-role quotient. Move each cohesive role class into a subsystem object or mixin with a shared context record.\n# The facade should expose public operations and delegate to the composed role authorities.'
            ),
            metrics=OrchestrationMetrics(
                function_line_count=0,
                branch_site_count=role_candidate.private_method_count,
                call_site_count=role_candidate.self_call_count,
                parameter_count=role_candidate.self_state_attribute_count,
                callee_family_count=len(role_candidate.role_method_counts),
            ),
        )
PassThroughCompositionFacadeCandidate = product_record('PassThroughCompositionFacadeCandidate', 'base_names: tuple[str, ...]', bases=(ClassLineWitnessCandidate,))
def _is_pass_through_class_body(body: Sequence[ast.stmt]) -> bool:
    trimmed = _trim_docstring_body(list(body))
    if not trimmed: return True
    return all((isinstance(statement, ast.Pass) or (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and (statement.value.value is Ellipsis)) for statement in trimmed))
def _pass_through_composition_facade_candidates(module: ParsedModule) -> tuple[PassThroughCompositionFacadeCandidate, ...]:
    candidates: list[PassThroughCompositionFacadeCandidate] = []
    for class_node in (
        node for node in ast.walk(module.module) if isinstance(node, ast.ClassDef)
    ):
        if len(class_node.bases) < 2: continue
        if not _is_pass_through_class_body(class_node.body): continue
        base_names = tuple(ast.unparse(base) for base in class_node.bases)
        candidates.append(PassThroughCompositionFacadeCandidate(file_path=str(module.path), line=class_node.lineno, class_name=class_node.name, base_names=base_names))
    return sorted_tuple(candidates, key=lambda item: (item.file_path, item.line, item.class_name))
declare_module_detector(PassThroughCompositionFacadeCandidate, high_confidence_certified_spec(PatternId.NOMINAL_STRATEGY_FAMILY, 'Pass-through composition facade should be derived from a composite spec', 'A class whose whole body is pass-through inheritance is a declaration of a composite family, not hand-written behavior. The architecture should name that composition as data and derive the class object from one generic composite-class authority.', 'generic composite-class derivation for pass-through multiple-inheritance facades', 'class body contains no behavior beyond composing several base roles', _NOMINAL_IDENTITY_MRO_ORDERING_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS, _CLASS_FAMILY_METHOD_ROLE_PARTIAL_VIEW_OBSERVATION_TAGS), CandidateFindingRenderer[PassThroughCompositionFacadeCandidate](summary=lambda facade_candidate: f"`{facade_candidate.class_name}` is a pass-through composition facade over {', '.join(facade_candidate.base_names)}.", evidence=lambda facade_candidate: (facade_candidate.evidence,), scaffold=lambda facade_candidate: '@dataclass(frozen=True)\nclass CompositeClassSpec:\n    name: str\n    bases: tuple[type, ...]\n    def build(self, module_name: str) -> type: ...', codemod_patch=lambda facade_candidate: f'# Replace `{facade_candidate.class_name}` with a derived class from a CompositeClassSpec.\n# Keep the ordered base list as data; derive the nominal facade class object generically.', metrics=lambda facade_candidate: HierarchyCandidateMetrics(duplicate_group_count=1, class_count=len(facade_candidate.base_names))), candidate_collector=_pass_through_composition_facade_candidates)
@dataclass(frozen=True)
class ProjectionPropertyFamilyCandidate(ClassLineWitnessCandidate):
    property_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    base_names: tuple[str, ...]
    @property
    def evidence_locations(self) -> tuple[SourceLocation, ...]: return tuple((SourceLocation(self.file_path, line, f'{self.class_name}.{property_name}') for line, property_name in zip(self.line_numbers, self.property_names, strict=True)))
def _is_self_attribute(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and (node.value.id == 'self'): return node.attr
    return None
def _is_path_projection_part(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str): return True
    if isinstance(node, ast.JoinedStr): return all((isinstance(value, ast.Constant) or (isinstance(value, ast.FormattedValue) and _is_self_attribute(value.value) is not None) for value in node.values))
    return False
def _path_projection_base(returned: ast.AST) -> str | None:
    node = returned
    saw_path_part = False
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        if not _is_path_projection_part(node.right): return None
        saw_path_part = True
        node = node.left
    if not saw_path_part: return None
    return _is_self_attribute(node)
def _projection_property_family_candidates(module: ParsedModule) -> tuple[ProjectionPropertyFamilyCandidate, ...]:
    candidates: list[ProjectionPropertyFamilyCandidate] = []
    for class_node in (
        node for node in ast.walk(module.module) if isinstance(node, ast.ClassDef)
    ):
        properties: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str]] = []
        for statement in class_node.body:
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)): continue
            if not any((_ast_terminal_name(decorator) == 'property' for decorator in statement.decorator_list)): continue
            body = _trim_docstring_body(statement.body)
            if len(body) != 1 or not isinstance(body[0], ast.Return): continue
            base_name = _path_projection_base(body[0].value)
            if base_name is None: continue
            properties.append((statement, base_name))
        if len(properties) < 3: continue
        ordered = sorted_tuple(properties, key=lambda item: item[0].lineno)
        candidates.append(ProjectionPropertyFamilyCandidate(file_path=str(module.path), line=class_node.lineno, class_name=class_node.name, property_names=tuple((function.name for function, _ in ordered)), line_numbers=tuple((function.lineno for function, _ in ordered)), base_names=sorted_tuple({base_name for _, base_name in ordered})))
    return sorted_tuple(candidates, key=lambda item: (item.file_path, item.line, item.class_name))
declare_module_detector(ProjectionPropertyFamilyCandidate, high_confidence_certified_spec(PatternId.DESCRIPTOR_DERIVED_VIEW, 'Path projection properties should be derived descriptors', 'Several properties project Path-valued views from owned base fields through the same `/` algebra. That is a descriptor-derived view family: the varying suffixes should be data while the projection algorithm lives in one reusable descriptor.', 'single descriptor authority for repeated Path projection properties', 'same class repeats Path projection properties over owned base fields', _AUTHORITATIVE_PROVENANCE_UNIT_RATE_COHERENCE_CAPABILITY_TAGS, _PROJECTION_HELPER_NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS), CandidateFindingRenderer[ProjectionPropertyFamilyCandidate](summary=lambda projection_candidate: f"`{projection_candidate.class_name}` repeats Path projection properties {', '.join(projection_candidate.property_names)} over bases {', '.join(projection_candidate.base_names)}.", evidence=lambda projection_candidate: projection_candidate.evidence_locations, scaffold=lambda projection_candidate: '@dataclass(frozen=True)\nclass PathProjection:\n    base_attr: str\n    parts: tuple[str, ...]\n    def __get__(self, instance, owner=None) -> Path: ...', codemod_patch=lambda projection_candidate: '# Replace repeated @property path projections with PathProjection descriptors.\n# Keep only base attribute and path parts as declarative data.', metrics=lambda projection_candidate: MappingMetrics(mapping_site_count=len(projection_candidate.property_names), field_count=len(projection_candidate.base_names), mapping_name=f'{projection_candidate.class_name} path projection', field_names=projection_candidate.property_names, source_name=', '.join(projection_candidate.base_names))), candidate_collector=_projection_property_family_candidates)
@dataclass(frozen=True)
class LiveTemplatePayloadFamilyCandidate(ClassLineWitnessCandidate):
    method_names: tuple[str, ...]
    line_numbers: tuple[int, ...]
    @property
    def evidence_locations(self) -> tuple[SourceLocation, ...]: return tuple((SourceLocation(self.file_path, line, f'{self.class_name}.{method_name}') for line, method_name in zip(self.line_numbers, self.method_names, strict=True)))
def _returns_direct_text_template(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    body = _trim_docstring_body(function.body)
    if len(body) != 1 or not isinstance(body[0], ast.Return): return False
    returned = body[0].value
    return (
        isinstance(returned, ast.JoinedStr)
        or (isinstance(returned, ast.Constant) and isinstance(returned.value, str))
    )
def _live_template_payload_family_candidates(module: ParsedModule) -> tuple[LiveTemplatePayloadFamilyCandidate, ...]:
    candidates: list[LiveTemplatePayloadFamilyCandidate] = []
    for class_node in (
        node for node in ast.walk(module.module) if isinstance(node, ast.ClassDef)
    ):
        template_methods = tuple((statement for statement in class_node.body if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and _returns_direct_text_template(statement)))
        if len(template_methods) < 3: continue
        ordered = sorted_tuple(template_methods, key=lambda item: item.lineno)
        candidates.append(LiveTemplatePayloadFamilyCandidate(file_path=str(module.path), line=class_node.lineno, class_name=class_node.name, method_names=tuple((method.name for method in ordered)), line_numbers=tuple((method.lineno for method in ordered))))
    return sorted_tuple(candidates, key=lambda item: (item.file_path, item.line, item.class_name))
declare_module_detector(LiveTemplatePayloadFamilyCandidate, high_confidence_certified_spec(PatternId.AUTHORITATIVE_SCHEMA, 'Live template payload methods should be derived from template specs', 'Several live methods return template payloads directly. Unlike dead payload emitters, these methods are real API, so the correct collapse is to keep the template declarations as data and derive the method surface from one generic template descriptor.', 'single template-method descriptor authority for live text payload families', 'same class repeats direct text-template return methods', _AUTHORITATIVE_PROVENANCE_UNIT_RATE_COHERENCE_CAPABILITY_TAGS, _EXPORT_NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS), CandidateFindingRenderer[LiveTemplatePayloadFamilyCandidate](summary=lambda template_candidate: f"`{template_candidate.class_name}` exposes live template payload methods {', '.join(template_candidate.method_names)}.", evidence=lambda template_candidate: template_candidate.evidence_locations, scaffold=lambda template_candidate: '@dataclass(frozen=True)\nclass TextTemplateMethod:\n    parameters: tuple[str, ...]\n    template: str\n    def __get__(self, instance, owner=None): ...', codemod_patch=lambda template_candidate: '# Replace direct template-return methods with TextTemplateMethod descriptors.\n# Keep template bodies declarative; derive the bound method API generically.', metrics=lambda template_candidate: MappingMetrics.from_field_names(mapping_site_count=len(template_candidate.method_names), mapping_name=f'{template_candidate.class_name} templates', field_names=template_candidate.method_names)), candidate_collector=_live_template_payload_family_candidates)
class PrivateCohortShouldBeModuleDetector(ConfiguredModuleCollectorCandidateDetector[PrivateCohortShouldBeModuleCandidate]):
    finding_spec = high_confidence_spec(PatternId.STAGED_ORCHESTRATION, 'Private subsystem cohort wants its own module', 'One module is carrying a tightly-coupled private subsystem cohort as if it were a whole package. The architecture wants a dedicated module for that bounded context, with the original file reduced to orchestration or public entry points.', 'explicit module-level subsystem boundaries with extracted private cohorts', 'one file contains a dense private context/result/helper family that should move together', _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS)
    def _finding_for_candidate(self, cohort: PrivateCohortShouldBeModuleCandidate) -> RefactorFinding:
        shared_tokens = ", ".join(cohort.shared_tokens[:3]) or "subsystem"
        sample_symbols = ', '.join((symbol.symbol for symbol in sorted(cohort.symbols, key=lambda item: (-item.line_count, item.line, item.symbol))[:3]))
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
class ParameterThreadFamilyDetector(ConfiguredModuleCollectorCandidateDetector[ParameterThreadFamilyCandidate]):
    finding_spec = high_confidence_spec(PatternId.AUTHORITATIVE_CONTEXT, 'Repeated threaded semantic parameter family', 'Several helpers keep re-threading the same semantic parameter bundle instead of carrying one nominal context. That weakens provenance and makes each helper signature a partially duplicated view of the same authority.', 'one authoritative context/request record for a shared semantic parameter family', 'the same semantic parameter bundle is threaded through several sibling helpers', _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS)
    def _finding_for_candidate(self, parameter_family: ParameterThreadFamilyCandidate) -> RefactorFinding: function_names = tuple((item.qualname for item in parameter_family.functions)); return self.build_finding(f"Functions {', '.join(function_names[:4])} thread the same semantic parameter family `{', '.join(parameter_family.shared_parameter_names)}` across {len(parameter_family.functions)} helpers.", tuple((item.evidence for item in parameter_family.functions[:6])), scaffold=_authoritative_context_scaffold(parameter_family), codemod_patch=_authoritative_context_patch(parameter_family), metrics=ParameterThreadMetrics(function_count=len(parameter_family.functions), shared_parameter_count=len(parameter_family.shared_parameter_names), shared_parameter_names=parameter_family.shared_parameter_names))
class SuffixAxisCompatibilitySurfaceDetector(ConfiguredModuleCollectorCandidateDetector[SuffixAxisSurfaceCandidate]):
    candidate_collector = _suffix_axis_surface_candidates
    finding_spec = high_confidence_spec(PatternId.AUTHORITATIVE_CONTEXT, 'Mirrored suffix-axis APIs should collapse to one authoritative context', 'Several operations are exposed once per suffix-named axis, such as `*_for_context` and `*_for_session`. When the same axis split repeats across an owner, the code is usually maintaining adapter surfaces instead of choosing one authoritative request/context record and deriving any compatibility projection at the boundary.', 'single authoritative context/request record instead of repeated suffix-axis adapter surfaces', 'same owner repeats an operation family across the same suffix-named axes', _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS, _METHOD_ROLE_PARTIAL_VIEW_NORMALIZED_AST_OBSERVATION_TAGS)
    def _finding_for_candidate(self, surface_candidate: SuffixAxisSurfaceCandidate) -> RefactorFinding:
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
                '@dataclass(frozen=True)\nclass OperationContext:\n    ...\n\n# Route operations through one authoritative context/session/request record.\n# Keep at most one boundary adapter that constructs the authority, not one adapter per operation.'
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
class SiblingRoleHelperSymmetryDetector(ModuleCollectorCandidateDetector[SiblingRoleHelperSymmetryCandidate]):
    finding_spec = high_confidence_spec(PatternId.LOCAL_VALUE_AUTHORITY, 'Sibling role helpers should collapse to one local authority', 'One owner has private helpers whose names differ by a role token but whose control skeletons and parameters are parallel. That is usually one local computation split into symmetrical role-specific helpers, which makes future changes require duplicated edits.', 'one authoritative local computation instead of parallel role-specific helpers', 'same owner has role-token sibling helpers with matching control skeletons', _AUTHORITATIVE_SHARED_ALGORITHM_AUTHORITY_PROVENANCE_CAPABILITY_TAGS, _METHOD_ROLE_NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS)
    def _finding_for_candidate(self, helper_candidate: SiblingRoleHelperSymmetryCandidate) -> RefactorFinding:
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
                f'def resolve_{shared_summary}(...):\n    # Compute the role-specific values together while the branch facts are live.\n    ...\n    return left_value, right_value\n\n# Use a small record only if this result crosses a boundary; keep local-only pairs as values.'
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
declare_module_detector(EnumStrategyDispatchCandidate, high_confidence_spec(PatternId.NOMINAL_STRATEGY_FAMILY, 'Enum strategy ladder wants nominal family', 'A closed enum/member dispatch ladder is choosing among behavior implementations inline. That wants an ABC-backed strategy family so each implementation guarantees one common method and the caller stops branching.', 'nominal strategy family with one guaranteed call surface', 'one owner branches over a closed enum/member family instead of delegating to implementation classes', _CLOSED_FAMILY_DISPATCH_NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_CAPABILITY_TAGS), CandidateFindingRenderer[EnumStrategyDispatchCandidate](summary=lambda dispatch_candidate: f"`{dispatch_candidate.qualname}` branches on `{dispatch_candidate.dispatch_axis}` across closed cases {', '.join(dispatch_candidate.case_names)} and should delegate to a nominal strategy family.", evidence=lambda dispatch_candidate: (dispatch_candidate.evidence,), scaffold=lambda dispatch_candidate: _nominal_strategy_scaffold(dispatch_candidate), codemod_patch=lambda dispatch_candidate: _nominal_strategy_patch(dispatch_candidate), metrics=lambda dispatch_candidate: DispatchCountMetrics.from_literal_family(dispatch_axis=dispatch_candidate.dispatch_axis, literal_cases=dispatch_candidate.case_names)), candidate_collector=_enum_strategy_dispatch_candidates)
class ResidualClosedAxisIndirectionDetector(ModuleCollectorCandidateDetector[ResidualClosedAxisIndirectionCandidate]):
    finding_spec = high_confidence_spec(PatternId.NOMINAL_STRATEGY_FAMILY, 'Enum-keyed table with residual branching should become a nominal strategy family', 'A function that indexes an enum-keyed table and still branches on the same enum axis is not using the table as an authority. The table is a degenerate projection over behavior that still lives in branches. The stronger normal form is an ABC-backed strategy family keyed by the enum, with `AutoRegisterMeta` owning import-time registration and any table-like views derived from the family.', 'metaclass-registry-backed nominal strategy family instead of enum table plus residual branching', 'same function indexes an enum-keyed table and branches on that enum axis', _AUTHORITATIVE_DISPATCH_CLOSED_FAMILY_DISPATCH_NOMINAL_IDENTITY_CAPABILITY_TAGS, _PROJECTION_DICT_BRANCH_DISPATCH_CLOSED_FAMILY_CASES_OBSERVATION_TAGS)
    def _finding_for_candidate(self, axis_candidate: ResidualClosedAxisIndirectionCandidate) -> RefactorFinding:
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
class RepeatedEnumStrategyDispatchDetector(ModuleCollectorCandidateDetector[RepeatedEnumStrategyDispatchCandidate]):
    finding_spec = high_confidence_spec(PatternId.NOMINAL_STRATEGY_FAMILY, 'Repeated closed-strategy dispatch should centralize in one nominal strategy family', 'Several owners re-dispatch the same closed enum family inline. The docs treat that as duplicated strategy orchestration: dispatch should happen once through one authoritative nominal strategy family or one shared strategy substrate.', 'single authoritative nominal strategy family for a repeated closed dispatch axis', 'same closed enum family is re-dispatched across sibling functions or methods', _CLOSED_FAMILY_DISPATCH_AUTHORITATIVE_DISPATCH_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS)
    def _finding_for_candidate(self, dispatch_candidate: RepeatedEnumStrategyDispatchCandidate) -> RefactorFinding: evidence = tuple((item.evidence for item in dispatch_candidate.functions[:6])); representative = dispatch_candidate.functions[0]; function_names = ', '.join((item.qualname for item in dispatch_candidate.functions[:4])); return self.build_finding(f"Functions {function_names} each re-dispatch `{dispatch_candidate.enum_family}` cases {', '.join(dispatch_candidate.shared_case_names)} inline.", evidence, scaffold=_nominal_strategy_scaffold(representative), codemod_patch=_nominal_strategy_patch(representative), metrics=DispatchCountMetrics(dispatch_site_count=len(dispatch_candidate.functions), dispatch_axis=dispatch_candidate.enum_family, literal_cases=dispatch_candidate.shared_case_names))
class InlineEnumSubsetGuardDetector(ModuleCollectorCandidateDetector[InlineEnumSubsetGuardCandidate]):
    finding_spec = high_confidence_spec(PatternId.AUTHORITATIVE_SCHEMA, 'Inline enum subset guard should derive from enum-owned policy', 'A branch that hardcodes an enum-member subset is a closed-axis policy table in disguise. The policy should be owned by the enum member or a typed row family, with any lookup derived exhaustively from that type-safe source.', 'type-safe enum-owned policy instead of inline enum subset literals', 'function branches on a hand-enumerated subset of one closed enum axis', _CLOSED_FAMILY_DISPATCH_AUTHORITATIVE_NOMINAL_IDENTITY_CAPABILITY_TAGS, _BRANCH_DISPATCH_PROJECTION_DICT_OBSERVATION_TAGS)
    def _finding_for_candidate(self, guard_candidate: InlineEnumSubsetGuardCandidate) -> RefactorFinding:
        cases = ', '.join((f'{guard_candidate.enum_name}.{case_name}' for case_name in guard_candidate.case_names))
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
class SplitDispatchAuthorityDetector(ModuleCollectorCandidateDetector[SplitDispatchAuthorityCandidate]):
    finding_spec = high_confidence_spec(PatternId.NOMINAL_STRATEGY_FAMILY, 'Cooperating dispatch layers should collapse into one product-family authority', 'The docs treat repeated cooperating dispatch layers as split authority. When one orchestration function selects a strategy-family implementation and separately routes another axis through `singledispatch`, the operation usually wants one authoritative product-family policy or one request-dispatched plan.', 'single authoritative product-family or request-dispatched policy for cooperating dispatch axes', 'one orchestrator combines a strategy-family selector with a separate singledispatch generic', _AUTHORITATIVE_DISPATCH_NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS, _CLASS_FAMILY_FACTORY_DISPATCH_REPEATED_METHOD_ROLES_OBSERVATION_TAGS)
    def _finding_for_candidate(self, dispatch_candidate: SplitDispatchAuthorityCandidate) -> RefactorFinding:
        evidence = (dispatch_candidate.evidence, SourceLocation(dispatch_candidate.file_path, dispatch_candidate.selector_line, f'{dispatch_candidate.strategy_root_name}.{dispatch_candidate.selector_method_name}'), SourceLocation(dispatch_candidate.file_path, dispatch_candidate.generic_line, dispatch_candidate.generic_function_name))
        return self.build_finding(
            (
                f"`{dispatch_candidate.qualname}` combines strategy selector "
                f"`{dispatch_candidate.strategy_root_name}.{dispatch_candidate.selector_method_name}({dispatch_candidate.strategy_axis_expression})` "
                f"with singledispatch `{dispatch_candidate.generic_function_name}({dispatch_candidate.generic_axis_expression})` "
                f"through callback `{dispatch_candidate.bridge_callback_name}`, splitting one operation across two dispatch authorities."
            ),
            evidence,
            scaffold=(
                '@dataclass(frozen=True)\nclass DispatchPlan:\n    strategy: object\n    source_type: type[object]\n\nclass ProductPolicy(ABC):\n    plan_key: ClassVar[DispatchPlan]\n    def run(self, request): ...\n'
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
                literal_cases=(*dispatch_candidate.strategy_case_names[:3], *dispatch_candidate.generic_case_names[:3]),
            ),
        )
class EmptyLeafProductFamilyDetector(ModuleCollectorCandidateDetector[EmptyLeafProductFamilyCandidate]):
    finding_spec = high_confidence_spec(PatternId.CLOSED_FAMILY_DISPATCH, 'Empty multiple-inheritance leaves should collapse into one product-family authority', 'The docs allow mixins for orthogonal reusable concerns, but empty leaf classes that merely enumerate all combinations of two reusable axes are usually a handwritten product table in inheritance form. That product should become one keyed authority or one product-family selector.', 'single authoritative keyed product family instead of empty inheritance combinations', 'empty leaf classes encode the full Cartesian product of two reusable inheritance axes', _AUTHORITATIVE_DISPATCH_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS, _CLASS_FAMILY_REPEATED_METHOD_ROLES_OBSERVATION_TAGS)
    def _finding_for_candidate(self, product_candidate: EmptyLeafProductFamilyCandidate) -> RefactorFinding:
        left_axis = ", ".join(product_candidate.left_axis_base_names)
        right_axis = ", ".join(product_candidate.right_axis_base_names)
        leaf_preview = ", ".join(product_candidate.leaf_class_names[:6])
        return self.build_finding(
            (
                f"Empty leaf classes {leaf_preview} encode `{left_axis}` x `{right_axis}` through multiple inheritance instead of one product-family authority."
            ),
            product_candidate.evidence,
            scaffold=(
                '@dataclass(frozen=True)\nclass ProductRule:\n    axis_left: object\n    axis_right: object\n    policy_type: type[object]\n\nPRODUCT_RULES = (...)\n'
            ),
            codemod_patch=(
                '# Replace the empty Cartesian-product leaf classes with one keyed product table or one nominal selector family.\n# Keep only irreducible axis-local behavior on the reusable bases; do not encode the cross product as `pass` subclasses.'
            ),
            metrics=DispatchCountMetrics.from_literal_family(
                dispatch_axis=(
                    f"{' | '.join(product_candidate.left_axis_base_names)} x {' | '.join(product_candidate.right_axis_base_names)}"
                ),
                literal_cases=product_candidate.leaf_class_names,
            ),
        )
class ClosedConstantSelectorDetector(ModuleCollectorCandidateDetector[ClosedConstantSelectorCandidate]):
    finding_spec = high_confidence_spec(PatternId.AUTHORITATIVE_SCHEMA, 'Closed selector over sibling constants should derive from one selector table', 'The docs treat branch ladders that choose among sibling specs, plans, contracts, or other immutable constants as duplicated selector logic once the constant family already exists. The selector should collapse into one authoritative keyed table or selector record so wrappers and downstream views are derived.', 'single authoritative selector table for a closed constant family', 'one function branches over a small predicate family and returns sibling constants or one shared wrapper around them', _AUTHORITATIVE_CLOSED_FAMILY_DISPATCH_PROVENANCE_CAPABILITY_TAGS, _BUILDER_CALL_DATAFLOW_ROOT_PREDICATE_CHAIN_OBSERVATION_TAGS)
    def _finding_for_candidate(self, selector_candidate: ClosedConstantSelectorCandidate) -> RefactorFinding:
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
                '@dataclass(frozen=True)\nclass SelectorRule:\n    key: object\n    selected: object\n\nSELECTOR_RULES = (\n    SelectorRule(key=..., selected=...),\n)\n_SELECTED_BY_KEY = {rule.key: rule.selected for rule in SELECTOR_RULES}\n'
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
class DerivedWrapperSpecShadowDetector(ModuleCollectorCandidateDetector[DerivedWrapperSpecShadowCandidate]):
    finding_spec = high_confidence_spec(PatternId.AUTHORITATIVE_SCHEMA, 'Generated wrapper spec family should collapse into the authoritative spec family', 'The docs treat writable wrapper-spec tables as secondary authorities when they just point back at an existing spec family and feed code generation. Wrapper metadata should live on the authoritative spec records so generated wrappers are derived from one source rather than synchronized across parallel tables.', 'single authoritative spec family carrying wrapper-generation metadata', 'secondary spec table references an authoritative spec family entry-by-entry and is only consumed by wrapper generation', _AUTHORITATIVE_PROVENANCE_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS, _BUILDER_CALL_DATAFLOW_ROOT_SCOPED_SHAPE_WRAPPER_OBSERVATION_TAGS)
    def _finding_for_candidate(self, shadow_candidate: DerivedWrapperSpecShadowCandidate) -> RefactorFinding:
        primary_family_label = (
            shadow_candidate.primary_family_name or shadow_candidate.primary_constructor_name
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
                '@dataclass(frozen=True)\nclass ExecutionSpec:\n    key: object\n    runner: object\n    wrapper_name: str | None = None\n    wrapper_defaults: dict[str, object] = field(default_factory=dict)\n\ndef build_wrapper(spec: ExecutionSpec): ...\n'
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
declare_module_detector(ModuleKeyedSelectionHelperCandidate, high_confidence_spec(PatternId.AUTHORITATIVE_SCHEMA, 'Local keyed-selection helper should collapse into the generic keyed-record table', 'The docs push reusable table/index machinery into one authoritative substrate. When a module defines a local selection-rule dataclass, a dict-index builder, and a keyed lookup helper that power multiple rule tables, it is reintroducing a second keyed-table framework instead of reusing the generic keyed-record helper.', 'single authoritative keyed-record table substrate reused across module-level selector tables', 'module-local selection helper framework powers multiple keyed rule tables', _AUTHORITATIVE_CLOSED_FAMILY_DISPATCH_PROVENANCE_CAPABILITY_TAGS, _BUILDER_CALL_DATAFLOW_ROOT_CLASS_FAMILY_OBSERVATION_TAGS), CandidateFindingRenderer[ModuleKeyedSelectionHelperCandidate](summary=lambda helper_candidate: f"`{helper_candidate.rule_class_name}`, `{helper_candidate.helper_function_name}`, and `{helper_candidate.lookup_function_name}` implement a local keyed-selection substrate for {', '.join(helper_candidate.rule_table_names[:4])} and indexes {', '.join(helper_candidate.index_table_names[:4])}.", evidence=lambda helper_candidate: helper_candidate.evidence, scaffold=lambda helper_candidate: 'KeyT = TypeVar("KeyT")\nRecordT = TypeVar("RecordT")\n\n@dataclass(frozen=True)\nclass KeyedRecordTable(Generic[KeyT, RecordT]):\n    records: tuple[RecordT, ...]\n    key_of: Callable[[RecordT], KeyT]\n    def require(self, key: KeyT, *, missing_error=None) -> RecordT: ...\n', codemod_patch=lambda helper_candidate: f'# Remove local keyed-selection helper `{helper_candidate.rule_class_name}` / `{helper_candidate.helper_function_name}` / `{helper_candidate.lookup_function_name}`.\n# Re-express these rule tables through the shared KeyedRecordTable substrate.', metrics=lambda helper_candidate: MappingMetrics(mapping_site_count=len(helper_candidate.rule_table_names), field_count=1, mapping_name=helper_candidate.rule_class_name, field_names=(helper_candidate.selected_field_name,), source_name=helper_candidate.helper_function_name, identity_field_names=('key',))), candidate_collector=_module_keyed_selection_helper_candidates)
declare_module_detector(CrossModuleAxisShadowFamilyCandidate, high_confidence_spec(PatternId.NOMINAL_STRATEGY_FAMILY, 'Cross-module shadow family should collapse into one axis authority', 'The docs require one authoritative owner per closed semantic axis. When one module already owns an enum/keyed family nominally and another module reintroduces a second family over the same cases, the axis has split authority and local behavior should derive from the authoritative family instead.', 'single authoritative closed-axis family reused across modules', 'same keyed enum axis is modeled by an authoritative family in one module and a shadow selector family in another', _AUTHORITATIVE_DISPATCH_NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS, _CLASS_FAMILY_FACTORY_DISPATCH_DATAFLOW_ROOT_OBSERVATION_TAGS), CandidateFindingRenderer[CrossModuleAxisShadowFamilyCandidate](summary=lambda shadow_candidate: f"Axis `{shadow_candidate.key_type_name}` is already owned by `{shadow_candidate.authoritative.family_name}` but re-encoded by `{shadow_candidate.shadow.family_name}.{shadow_candidate.selector_method_name}` across cases {', '.join(shadow_candidate.shared_case_names[:4])}.", evidence=lambda shadow_candidate: shadow_candidate.evidence, scaffold=lambda shadow_candidate: _axis_policy_registry_scaffold('invariant(self)') + f'\n\ndef run_with_axis(axis: {_AXIS_POLICY_KEY_TYPE_NAME}, ...):\n    policy = {_AXIS_POLICY_ROOT_NAME}.for_key(axis)\n    # derive local execution from authoritative policy facts\n', codemod_patch=lambda shadow_candidate: f'# Remove shadow family `{shadow_candidate.shadow.family_name}`.\n# Derive local behavior from authoritative family `{shadow_candidate.authoritative.family_name}` instead of re-owning axis `{shadow_candidate.key_type_name}`.', metrics=lambda shadow_candidate: _axis_dispatch_metrics(shadow_candidate.shared_case_names, shadow_candidate.key_type_name)), detector_base=CrossModuleCollectorCandidateDetector, candidate_collector=_cross_module_axis_shadow_family_candidates)
class ResidualClosedAxisBranchingDetector(CrossModuleCollectorCandidateDetector[ResidualClosedAxisBranchingCandidate]):
    finding_spec = high_confidence_spec(PatternId.CLOSED_FAMILY_DISPATCH, 'Manual closed-axis branching should derive from existing keyed authority', 'The docs require one authoritative owner per closed enum/key axis. When a keyed nominal family already owns that axis, downstream `if`/`match` ladders over the same cases become residual shadow dispatch.', 'behavior derived from authoritative keyed family rather than downstream enum branching', 'function branches on an enum axis already owned by a keyed nominal family in another module', _AUTHORITATIVE_DISPATCH_CLOSED_FAMILY_DISPATCH_NOMINAL_IDENTITY_CAPABILITY_TAGS, _BRANCH_DISPATCH_CLASS_FAMILY_DATAFLOW_ROOT_OBSERVATION_TAGS)
    def _finding_for_candidate(self, residual_candidate: ResidualClosedAxisBranchingCandidate) -> RefactorFinding:
        authoritative_family_names = ', '.join((family_name for family_name, _, _ in residual_candidate.authoritative_families[:4]))
        return self.build_finding(
            (
                f"`{residual_candidate.qualname}` branches {residual_candidate.branch_site_count} time(s) on axis "
                f"`{residual_candidate.key_type_name}` across cases {', '.join(residual_candidate.case_names)}, "
                f"even though authoritative family `{authoritative_family_names}` already owns that axis."
            ),
            residual_candidate.evidence,
            scaffold=(
                _axis_policy_registry_scaffold("apply(self, context)")
                + f'\n\ndef run(context):\n    policy = {_AXIS_POLICY_ROOT_NAME}.for_key(context.axis)\n    return policy.apply(context)\n'
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
class ParallelKeyedAxisFamilyDetector(CrossModuleCollectorCandidateDetector[ParallelKeyedAxisFamilyCandidate]):
    finding_spec = high_confidence_spec(PatternId.NOMINAL_STRATEGY_FAMILY, 'Parallel keyed families should collapse into one axis authority', 'The docs require one authoritative nominal owner per closed semantic axis. When two modules each define a keyed family over the same enum/key cases, the axis has split ownership even if both sides are nominal.', 'single cross-module keyed-axis authority with module-local adapters derived from it', 'same keyed enum axis is modeled by multiple nominal families across modules', _AUTHORITATIVE_DISPATCH_AUTHORITATIVE_NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS, _CLASS_FAMILY_FACTORY_DISPATCH_DATAFLOW_ROOT_OBSERVATION_TAGS)
    def _finding_for_candidate(self, family_candidate: ParallelKeyedAxisFamilyCandidate) -> RefactorFinding:
        shared_cases = ", ".join(family_candidate.shared_case_names[:4])
        label_clause = ""
        if family_candidate.left.family_label is not None and family_candidate.left.family_label == family_candidate.right.family_label: label_clause = f' Both declare family label `{family_candidate.left.family_label}`.'
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
                + '\n\n# Keep one authoritative keyed family and let secondary modules derive local adapters/specs from it.'
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
declare_module_detector(ParallelKeyedTableAxisCandidate, high_confidence_spec(PatternId.AUTHORITATIVE_SCHEMA, 'Parallel enum-keyed tables across modules should collapse into one axis record', 'The docs require one authoritative writable owner per closed semantic axis. When multiple modules maintain separate enum-keyed tables over the same cases, the axis is split across parallel metadata maps.', 'single authoritative enum-keyed row family with derived module-local projections', 'same closed enum/key axis is encoded by multiple keyed tables across modules', _AUTHORITATIVE_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS, _PROJECTION_DICT_DATAFLOW_ROOT_BUILDER_CALL_OBSERVATION_TAGS), CandidateFindingRenderer[ParallelKeyedTableAxisCandidate](summary=lambda table_candidate: f"Axis `{table_candidate.key_type_name}` is restated by `{table_candidate.left.table_name}` and `{table_candidate.right.table_name}` across cases {', '.join(table_candidate.shared_case_names[:4])}.", evidence=lambda table_candidate: table_candidate.evidence, scaffold=lambda table_candidate: '@dataclass(frozen=True)\nclass AxisRow:\n    key: AxisEnum\n    primary: object\n    secondary: object | None = None\n\nAXIS_ROWS = (\n    AxisRow(key=AxisEnum.ALPHA, primary=..., secondary=...),\n)\nAXIS_ROW_BY_KEY = {row.key: row for row in AXIS_ROWS}\n', codemod_patch=lambda table_candidate: f'# Collapse `{table_candidate.left.table_name}` and `{table_candidate.right.table_name}` onto one authoritative row family.\n# Keep one writable axis table and derive any module-local indexes or views from it.', metrics=lambda table_candidate: MappingMetrics(mapping_site_count=2, field_count=max(len(table_candidate.shared_case_names), 1), mapping_name=table_candidate.left.table_name, field_names=table_candidate.shared_case_names, source_name=table_candidate.key_type_name, identity_field_names=('key',))), detector_base=CrossModuleCollectorCandidateDetector, candidate_collector=_parallel_keyed_table_axis_candidates)
class ParallelKeyedTableAndFamilyDetector(CrossModuleCollectorCandidateDetector[ParallelKeyedTableAndFamilyCandidate]):
    finding_spec = high_confidence_spec(PatternId.AUTHORITATIVE_SCHEMA, 'Keyed table and keyed family should collapse into one auto-registered axis family', 'The docs require one authoritative owner per closed semantic axis. When a module keeps one keyed table of per-case records and a second keyed nominal family over the same cases, the axis is split across data and behavior. If the family already carries the runtime behavior boundary, the table should derive from that family instead of competing with it.', 'single authoritative metaclass-registry axis family with derived table/view projections', 'same enum/key axis is encoded by both a keyed table and a keyed nominal family', _AUTHORITATIVE_AUTHORITATIVE_DISPATCH_NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS, _CLASS_FAMILY_BUILDER_CALL_DATAFLOW_ROOT_OBSERVATION_TAGS)
    def _finding_for_candidate(self, table_candidate: ParallelKeyedTableAndFamilyCandidate) -> RefactorFinding:
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
                + f'\n\n@dataclass(frozen=True)\nclass DerivedAxisRow:\n    key: {_AXIS_POLICY_KEY_TYPE_NAME}\n    policy_type: type[{_AXIS_POLICY_ROOT_NAME}]\n    config: object\n\ndef build_axis_rows() -> tuple[DerivedAxisRow, ...]:\n    return tuple(\n        DerivedAxisRow(key=key, policy_type=policy_type, config=...)\n        for key, policy_type in {_AXIS_POLICY_ROOT_NAME}.__registry__.items()\n    )'
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
class EnumKeyedTableClassAxisShadowDetector(ModuleCollectorCandidateDetector[EnumKeyedTableClassAxisShadowCandidate]):
    finding_spec = high_confidence_spec(PatternId.AUTHORITATIVE_SCHEMA, 'Enum-keyed table should derive from auto-registered class-declared axis keys', 'The docs require a single writable owner per closed semantic axis. If a module already declares that axis through class-level enum assignments, adding a writable enum-keyed table over the same cases creates duplicate authority and a synchronization surface. The class-declared axis should be the primary owner and any enum-keyed lookup should be derived from the family registry.', 'one authoritative metaclass-registry closed-axis owner with derived table/view projections', 'module-level enum-keyed table overlaps a class family that already declares the same enum axis', _AUTHORITATIVE_CLOSED_FAMILY_DISPATCH_NOMINAL_IDENTITY_CAPABILITY_TAGS, _PROJECTION_DICT_CLASS_FAMILY_DATAFLOW_ROOT_OBSERVATION_TAGS)
    def _finding_for_candidate(self, axis_candidate: EnumKeyedTableClassAxisShadowCandidate) -> RefactorFinding:
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
                + f'\n\nAXIS_BY_KEY = {{\n    key: policy_type\n    for key, policy_type in {_AXIS_POLICY_ROOT_NAME}.__registry__.items()\n}}\n'
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
class TransportShellTemplateMethodDetector(ConfiguredModuleCollectorCandidateDetector[TransportShellTemplateCandidate]):
    candidate_collector = _transport_shell_template_candidates
    finding_spec = high_confidence_spec(PatternId.ABC_TEMPLATE_METHOD, 'Template-method family is a transport shell over a downstream authority', 'The docs say nominal families should have one authoritative owner. When an ABC template method only materializes an intermediate object from a class-level selector, delegates through one hook, and repackages through another hook, the extra family is usually a transport shell around an already authoritative boundary.', 'single authoritative materialization/execution family instead of a parallel transport shell', 'template family varies mostly by class-level selector and result adapter', _AUTHORITATIVE_SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_CAPABILITY_TAGS, _CLASS_FAMILY_BUILDER_CALL_DATAFLOW_ROOT_OBSERVATION_TAGS)
    def _finding_for_candidate(self, shell_candidate: TransportShellTemplateCandidate) -> RefactorFinding:
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
                '@dataclass(frozen=True)\nclass MaterializationSpec:\n    selector: object\n    materializer: object\n    executor: object\n    packager: object\n# Dispatch once on the authoritative selector/spec family.'
            ),
            codemod_patch=(
                f"# Collapse `{shell_candidate.class_name}` onto the downstream selector/spec family.\n"
                "# Keep one selection boundary and let that boundary own materialization, execution, and result packaging."
            ),
        )
class CrossModuleSpecAxisAuthorityDetector(ConfiguredCrossModuleCollectorCandidateDetector[CrossModuleSpecAxisAuthorityCandidate]):
    finding_spec = high_confidence_spec(PatternId.AUTHORITATIVE_SCHEMA, 'Cross-module spec axis should have one authority', 'The docs say one semantic family should have one authoritative owner. When two modules encode the same identity-axis -> executable-axis spec pairs, one table is a duplicate authority unless it is explicitly derived.', 'one repository-wide authoritative spec-axis family', 'same identity/executable spec axis is re-encoded across modules', _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS, _BUILDER_CALL_DATAFLOW_ROOT_CLASS_FAMILY_OBSERVATION_TAGS)
    def _finding_for_candidate(self, authority_candidate: CrossModuleSpecAxisAuthorityCandidate) -> RefactorFinding:
        family_names = ', '.join((f'{Path(family.file_path).name}:{family.family_name}' for family in authority_candidate.families))
        pair_names = ', '.join((f'{identity}->{executable}' for identity, executable in authority_candidate.shared_axis_pairs))
        axis_fields = " -> ".join(authority_candidate.axis_field_names)
        evidence = tuple((family.evidence for family in authority_candidate.families[:6]))
        return self.build_finding(
            (
                f"Families {family_names} each encode the same `{axis_fields}` pairs {pair_names} across module boundaries."
            ),
            evidence,
            scaffold=(
                '@dataclass(frozen=True)\nclass AxisExecutionSpec:\n    identity: object\n    executable: object\n# Keep one exported authority and let downstream modules compose from it.'
            ),
            codemod_patch=(
                '# Extract one repository-wide spec-axis family.\n# Make downstream wrappers, benchmarks, or adapters reference that authority instead of restating identity/executable pairs.'
            ),
        )
class ParallelRegistryProjectionFamilyDetector(ModuleCollectorCandidateDetector[ParallelRegistryProjectionFamilyCandidate]):
    finding_spec = high_confidence_spec(PatternId.AUTHORITATIVE_SCHEMA, 'Parallel registry projection builders should collapse into one family spec', 'The docs say one semantic family should have one authoritative owner. When several functions differ only in which registry authority feeds which target constructor, the projection-axis mapping should become one declared spec or family authority instead of several hand-wired wrappers.', 'single authoritative registry-projection family', 'same registry-authority-to-target projection shape repeated across sibling functions', _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS, _BUILDER_CALL_CLASS_FAMILY_DATAFLOW_ROOT_OBSERVATION_TAGS)
    def _finding_for_candidate(self, catalog_candidate: ParallelRegistryProjectionFamilyCandidate) -> RefactorFinding:
        function_names = ', '.join((function.qualname for function in catalog_candidate.functions[:4]))
        extractor_bases = ', '.join((function.extractor_base_name for function in catalog_candidate.functions[:4]))
        catalog_types = ', '.join((function.catalog_type_name for function in catalog_candidate.functions[:4]))
        evidence = tuple(function.evidence for function in catalog_candidate.functions[:6])
        return self.build_finding(
            (
                f"Functions {function_names} each build {catalog_types} through "
                f"`{catalog_candidate.collector_name}(structure, ExtractorBase.{catalog_candidate.registry_accessor_name}())` "
                f"over parallel extractor bases {extractor_bases}."
            ),
            evidence,
            scaffold=(
                '@dataclass(frozen=True)\nclass RegistryProjectionSpec:\n    registry_authority: type\n    target_type: type\n# One helper should own the registry-authority to target mapping.'
            ),
            codemod_patch=(
                '# Extract one registry-projection family spec and one authoritative projection builder.\n# Make per-axis public helpers delegate to that authority instead of reconstructing collector(...registry_accessor()).'
            ),
        )
class RepeatedKeyedFamilyDetector(ConfiguredCrossModuleCollectorCandidateDetector[RepeatedKeyedFamilyCandidate]):
    finding_spec = high_confidence_spec(PatternId.AUTO_REGISTER_META, 'Repeated keyed family scaffolding should collapse into one typed metaclass-registry base', 'The docs encourage aggressive metaprogramming when several nominal families repeat the same class-level registration and lookup shell. When many roots restate `registry_key_attr`, `_registry`, and `for_*` lookup methods, the family algorithm should live in one typed `metaclass-registry` base.', 'single typed metaclass-registry substrate for keyed nominal registries', 'same keyed family registration and lookup shell repeated across nominal family roots', _CLASS_LEVEL_REGISTRATION_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS, _CLASS_FAMILY_DATAFLOW_ROOT_OBSERVATION_TAGS)
    def _finding_for_candidate(self, family_candidate: RepeatedKeyedFamilyCandidate) -> RefactorFinding:
        class_names = ', '.join((root.class_name for root in family_candidate.roots[:8]))
        lookup_names = ', '.join(sorted({root.lookup_method_name for root in family_candidate.roots[:8]}))
        registry_keys = ', '.join(sorted({root.registry_key_attr_name for root in family_candidate.roots[:8]}))
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
                '# Extract one typed metaclass-registry base that owns registration lookup, duplicate handling, and error shaping.\n# Leave only declarative key attributes and irreducible hook methods on each family root, and read the registered classes from `cls.__registry__`.'
            ),
        )
class ManualKeyedRecordTableDetector(ConfiguredModuleCollectorCandidateDetector[ManualKeyedRecordTableGroupCandidate]):
    candidate_collector = _manual_keyed_record_table_group_candidates
    finding_spec = high_confidence_spec(PatternId.AUTHORITATIVE_SCHEMA, 'Manual keyed record tables should collapse into one authoritative spec table', 'When several frozen record classes repeat `_registry`, `register`, and `for_*` lookup around closed keys, the code is hand-maintaining multiple writable tables. The docs prefer one authoritative spec tuple or generic keyed-record table with derived indexes.', 'single authoritative keyed-record table or derived index', 'same manual record registration and keyed lookup shell repeated across data classes', _AUTHORITATIVE_CLOSED_FAMILY_DISPATCH_PROVENANCE_CAPABILITY_TAGS, _BUILDER_CALL_DATAFLOW_ROOT_CLASS_FAMILY_OBSERVATION_TAGS)
    def _finding_for_candidate(self, group_candidate: ManualKeyedRecordTableGroupCandidate) -> RefactorFinding:
        class_names = ', '.join((item.class_name for item in group_candidate.classes[:6]))
        key_fields = ', '.join(sorted({item.key_field_name for item in group_candidate.classes[:6]}))
        lookup_names = ', '.join(sorted({item.lookup_method_name for item in group_candidate.classes[:6]}))
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
                '# Replace per-class mutable `_registry` + `register` shells with one authoritative tuple of record specs.\n# Derive the keyed lookup dict once, or factor the pattern into a generic keyed-record table helper.'
            ),
        )
class ManualStructuralRecordMechanicsDetector(ConfiguredModuleCollectorCandidateDetector[ManualStructuralRecordMechanicsGroupCandidate]):
    candidate_collector = _manual_structural_record_mechanics_group_candidates
    finding_spec = high_confidence_spec(PatternId.AUTHORITATIVE_SCHEMA, 'Repeated structural record mechanics should derive from field metadata', 'When several frozen dataclass records hand-write validation, tuple-style field projection, round-trip reconstruction, and fieldwise transform logic, those mechanics have become a second authority beside the field declarations. The docs prefer one metadata-driven record substrate that derives those mechanics from typed fields.', 'single typed structural-record substrate with derived validation, projection, and transform mechanics', 'same dataclass record lifecycle mechanics repeated across sibling structural record classes', _AUTHORITATIVE_FAIL_LOUD_CONTRACTS_PROVENANCE_TYPE_LINEAGE_CAPABILITY_TAGS, _CLASS_FAMILY_DATAFLOW_ROOT_BUILDER_CALL_OBSERVATION_TAGS)
    def _finding_for_candidate(self, group_candidate: ManualStructuralRecordMechanicsGroupCandidate) -> RefactorFinding:
        class_names = ', '.join((item.class_name for item in group_candidate.classes[:6]))
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
                '@dataclass_transform(field_specifiers=(field, record_field))\nclass StructuralRecordBase:\n    def validate(self): ...\n    def project_fields(self): ...\n    @classmethod\n    def from_projected(cls, projected, metadata): ...\n    def transformed(self, **changes): ...\n'
            ),
            codemod_patch=(
                '# Move validation constraints, projected-field partitions, and transform semantics into typed field metadata.\n# Derive projection, round-trip reconstruction, and fieldwise transforms from one structural-record base instead of re-encoding them per class.'
            ),
        )
class RepeatedConcreteTypeCaseAnalysisDetector(ConfiguredCrossModuleCollectorCandidateDetector[RepeatedConcreteTypeCaseAnalysisCandidate]):
    finding_spec = high_confidence_spec(PatternId.NOMINAL_INTERFACE_WITNESS, 'Repeated concrete-type recovery should become nominal family behavior', 'When several functions repeatedly recover the same semantic family through concrete `isinstance` checks on one carried attribute, the family boundary is still latent. The docs want one nominal ABC and concrete leaf behavior exposed through typed properties or hooks instead of repeated leaf decoding.', 'single ABC-backed family for the carried subject, with repeated case recovery moved into nominal properties or hooks', 'same attribute-carried family is re-decoded through repeated concrete runtime type checks across several functions', _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_MRO_ORDERING_CAPABILITY_TAGS, _CLASS_FAMILY_DATAFLOW_ROOT_PARTIAL_VIEW_OBSERVATION_TAGS)
    def _finding_for_candidate(self, case_candidate: RepeatedConcreteTypeCaseAnalysisCandidate) -> RefactorFinding:
        function_names = ', '.join((function.function_name for function in case_candidate.functions[:6]))
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
        return self.build_finding(
            (
                f"Functions {function_names} repeatedly recover `{case_candidate.subject_role}` across concrete classes {class_names}.{alias_summary}{existing_base_summary}"
            ),
            case_candidate.evidence,
            scaffold=(
                f'class {suggested_family_name}(ABC):\n    @property\n    @abstractmethod\n    def case_label(self) -> str: ...\n\n    def explain_case(self, context):\n        return None\n'
            ),
            codemod_patch=(
                f'# Type `{case_candidate.subject_role}` against one nominal ABC family instead of a concrete union surface.\n# Move repeated concrete `isinstance` recovery into abstract properties or case hooks on that family.\n# Keep only irreducible case-local residue in the concrete subclasses.'
            ),
            metrics=DispatchCountMetrics(
                dispatch_site_count=len(case_candidate.functions),
                dispatch_axis=case_candidate.subject_role,
                literal_cases=case_candidate.concrete_class_names,
            ),
        )
class ImplicitSelfContractMixinDetector(ConfiguredCrossModuleCollectorCandidateDetector[ImplicitSelfContractMixinCandidate]):
    finding_spec = high_confidence_spec(PatternId.ABC_TEMPLATE_METHOD, 'Concrete mixins should not hide consumer contracts behind `self`-casts', 'The docs reserve mixins for orthogonal reusable concerns that participate in nominal MRO cleanly. When a concrete mixin erases `self` through `cast(..., self)` to reach consumer-owned fields, the mixin is carrying non-orthogonal family logic through a hidden contract instead of a declared base or policy.', 'declared nominal base or policy row for the shared algorithm instead of a hidden mixin self-contract', 'concrete mixin methods erase `self` through casts and depend on consumer-owned attributes across several subclasses', _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS, _CLASS_FAMILY_REPEATED_METHOD_ROLES_PARTIAL_VIEW_OBSERVATION_TAGS)
    def _finding_for_candidate(self, mixin_candidate: ImplicitSelfContractMixinCandidate) -> RefactorFinding:
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
                'class FamilyBase(ABC):\n    def run_shared_step(self): ...\n\nclass CasePolicy(ABC):\n    def run(self, request): ...\n'
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
class RepeatedGuardValidatorFamilyDetector(ConfiguredModuleCollectorCandidateDetector[RepeatedGuardValidatorFamilyCandidate]):
    finding_spec = high_confidence_spec(PatternId.ABC_TEMPLATE_METHOD, 'Repeated guard validators should collapse into one case-policy authority', 'When several sibling boolean helpers walk the same subject through fail-fast guards and case-local final checks, the algorithm skeleton is split across helper names instead of being owned by one nominal case policy or declarative rule family.', 'single authoritative case-policy or rule-table validator', 'same subject and subordinate view validated through repeated fail-fast sibling helpers', _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_AUTHORITATIVE_CAPABILITY_TAGS, _DATAFLOW_ROOT_PARTIAL_VIEW_CLASS_FAMILY_OBSERVATION_TAGS)
    def _finding_for_candidate(self, family_candidate: RepeatedGuardValidatorFamilyCandidate) -> RefactorFinding:
        function_names = ', '.join((function.function_name for function in family_candidate.functions[:6]))
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
        return self.build_finding(
            (
                f"Boolean validators {function_names} each guard `{family_candidate.subject_param_name}`{alias_summary} "
                f"with the same fail-fast attribute checks over {shared_attrs}.{helper_summary}"
            ),
            family_candidate.evidence,
            scaffold=(
                'class ValidationCasePolicy(ABC):\n    def validation_error(self, subject):\n        child = self._subject_child(subject)\n        if not self._shared_preconditions(subject, child):\n            return self._shared_failure_message()\n        return self._case_specific_error(subject, child)\n\n    @abstractmethod\n    def _case_specific_error(self, subject, child): ...'
            ),
            codemod_patch=(
                '# Collapse these sibling boolean helpers into one authoritative case-policy family or one declarative rule table.\n# Keep shared fail-fast guards in one concrete validator method, and leave only case-specific predicates or handle sets per case.'
            ),
        )
class RepeatedValidateShapeGuardFamilyDetector(IssueDetector):
    finding_spec = high_confidence_spec(PatternId.ABC_TEMPLATE_METHOD, 'Repeated validate() shape guards should collapse into one validated-record authority', 'Sibling nominal records repeat the same fail-fast shape and dimensional guards in `validate()` while differing only in field names or a small residue check. The docs treat that as duplicated contract authority that should move into one shared validated-record base, field-spec table, or mixin hook.', 'single authoritative validated-record contract for repeated shape/ndim guards', 'same nominal record family repeats fail-loud shape validation scaffolding', _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_AUTHORITATIVE_CAPABILITY_TAGS, _CLASS_FAMILY_METHOD_ROLE_NORMALIZED_AST_OBSERVATION_TAGS)
    def _collect_findings(self, modules: list[ParsedModule], config: DetectorConfig) -> list[RefactorFinding]: return [self._finding_for_candidate(candidate) for candidate in _repeated_validate_shape_guard_candidates_for_modules(modules, config)]
    def _finding_for_candidate(self, candidate: object) -> RefactorFinding:
        family_candidate = cast(RepeatedValidateShapeGuardFamilyCandidate, candidate)
        method_symbols = tuple(method.symbol for method in family_candidate.methods)
        method_summary = ", ".join(method_symbols[:6])
        shared_guard_count = len(family_candidate.shared_shape_guard_signatures)
        shared_guard_preview = ', '.join(family_candidate.shared_shape_guard_signatures[:3])
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
                f'class ShapeValidatedRecord(ABC):\n    def validate(self):\n        for predicate, message in self._shape_guard_rules():\n            if predicate(self):\n                raise ValueError(message)\n        self._validate_residue()\n\n    @classmethod\n    @abstractmethod\n    def _shape_guard_rules(cls): ...\n\n    def _validate_residue(self):\n        return None{preview_suffix}'
            ),
            codemod_patch=(
                '# Collapse repeated `validate()` shape guards into one authoritative validated-record base or field-spec table.\n# Keep only the truly variable residue checks, messages, or field roster on each concrete record.'
            ),
        )
class RepeatedResultAssemblyPipelineDetector(ConfiguredModuleCollectorCandidateDetector[RepeatedResultAssemblyPipelineCandidate]):
    finding_spec = high_confidence_spec(PatternId.ABC_TEMPLATE_METHOD, 'Repeated result-assembly pipeline should collapse into one authoritative assembler', 'Several owners repeat the same downstream result-assembly stages and differ only in the upstream source or projection that feeds the pipeline. The docs treat that as shared algorithm authority that should move into one template method or authoritative helper with one orthogonal source hook.', 'single authoritative result-assembly pipeline with one source hook', 'same staged assembly tail is repeated across sibling functions or methods', _SHARED_ALGORITHM_AUTHORITY_AUTHORITATIVE_NOMINAL_IDENTITY_CAPABILITY_TAGS)
    def _finding_for_candidate(self, pipeline_candidate: RepeatedResultAssemblyPipelineCandidate) -> RefactorFinding:
        function_names = ', '.join((function.qualname for function in pipeline_candidate.functions[:4]))
        stage_names = ', '.join((stage.callee_name for stage in pipeline_candidate.shared_tail))
        evidence = tuple((function.evidence for function in pipeline_candidate.functions[:6]))
        return self.build_finding(
            (
                f"Functions {function_names} share the same result-assembly tail "
                f"{stage_names} and differ only in their leading source stages."
            ),
            evidence,
            scaffold=(
                'class ResultAssembler(ABC):\n    @abstractmethod\n    def supply_inputs(self, request): ...\n\n    def assemble(self, request):\n        supplied = self.supply_inputs(request)\n        # run the shared downstream assembly stages here\n        return result'
            ),
            codemod_patch=(
                '# Extract the shared assignment/return tail into one authoritative helper.\n# Leave only the source-supplier stage variant-specific.'
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
    def render(self) -> str: step_rows = '\n'.join((f'        {step_name}(),' for step_name in self.step_names)); return f'class {self.step_base_name}(EffectStep, ABC):\n    normal_form = {self.normal_form!r}\n\n@dataclass(frozen=True)\nclass {self.matcher_name}:\n    steps: tuple[{self.step_base_name}, ...] = (\n{step_rows}\n    )\n\n    def {self.method_name}(self, {self.input_name}):\n        return Maybe.of({self.input_name}).bind_all(self.steps)'
_DEFAULT_NORMAL_FORM_SCAFFOLD = (
    "class CandidateStep(EffectStep, ABC):\n    normal_form = 'typed_effect_carrier'\n\n@dataclass(frozen=True)\nclass CandidateMatcher:\n    steps: tuple[CandidateStep, ...] = (ExtractFirst(), ExtractSecond(), BuildWitness())\n\n    def build_candidate(self, source):\n        return Maybe.of(source).bind_all(self.steps)"
)
_NORMAL_FORM_SCAFFOLDS = {
    spec_name: _NormalFormScaffoldSpec(spec_name, matcher_name, method_name, input_name, step_base_name, steps)
    for spec_name, matcher_name, method_name, input_name, step_base_name, steps in (('ast_shape_matcher', 'AstShapeMatcher', 'match_shape', 'node', 'AstShapeMatcherStep', ('ExpectCall', 'ExpectSingleArgument', 'ExpectNamedAstShape', 'BuildWitness')), ('transport_call_chain_matcher', 'TransportChainMatcher', 'match_transport_chain', 'function', 'TransportChainMatcherStep', ('SingleReturnCall', 'CallChain', 'TransportedValues', 'BuildTransportWitness')), ('comparison_guard_matcher', 'ComparisonGuardMatcher', 'match_comparison_guard', 'test', 'ComparisonGuardMatcherStep', ('SingleCompare', 'EnumMemberPair', 'BuildGuardPolicy')), ('loop_fold_matcher', 'LoopFoldMatcher', 'match_loop_fold', 'body', 'LoopFoldMatcherStep', ('ExpectAssignment', 'ExpectLoop', 'ExpectReturnedAccumulator', 'BuildFoldWitness')), ('statement_sequence_matcher', 'StatementSequenceMatcher', 'match_sequence', 'function', 'StatementSequenceMatcherStep', ('ExpectRoleSequence', 'BuildWitness')))
}
class FailSoftEffectPipelineDetector(ConfiguredModuleCollectorCandidateDetector[FailSoftEffectPipelineCandidate]):
    finding_spec = finding_spec_template(PatternId.STAGED_ORCHESTRATION, 'Fail-soft optional pipeline should use a typed effect carrier', 'A function that repeatedly exits through `return None` is manually threading an optional effect through every extraction stage. The semantic-compressor normal form is a typed `Maybe`/`Result` carrier plus nominal matcher-step objects that own absence/provenance once instead of restating the guard at every stage.', 'single typed effect carrier with nominal inherited matcher steps for optional extraction, validation, and provenance flow', 'same fail-soft absence effect is manually re-threaded across one extraction pipeline', _SHARED_ALGORITHM_AUTHORITY_PROVENANCE_FAIL_LOUD_CONTRACTS_CAPABILITY_TAGS, _PREDICATE_CHAIN_DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS)
    _DEFAULT_NORMAL_FORM_SCAFFOLD = _DEFAULT_NORMAL_FORM_SCAFFOLD
    _NORMAL_FORM_SCAFFOLDS: ClassVar[dict[str, _NormalFormScaffoldSpec]] = (
        _NORMAL_FORM_SCAFFOLDS
    )
    def _normal_form_scaffold(self, normal_form: str) -> str: spec = self._NORMAL_FORM_SCAFFOLDS.get(normal_form); return spec.render() if spec is not None else self._DEFAULT_NORMAL_FORM_SCAFFOLD
    def _finding_for_candidate(self, pipeline_candidate: FailSoftEffectPipelineCandidate) -> RefactorFinding:
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
                f"normal form is `{pipeline_candidate.normal_form}`."
                f"{helper_suffix}"
            ),
            (pipeline_candidate.evidence,),
            scaffold=self._normal_form_scaffold(pipeline_candidate.normal_form),
            codemod_patch=(
                f"# Collapse repeated `if value is None: return None` guard stages into `{pipeline_candidate.normal_form}`.\n"
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
def _effect_step_payoff_scaffold(candidate: EffectStepAmortizationCandidate) -> str: normal_form_class = _camel_case(candidate.normal_form); return f"class {normal_form_class}Step(EffectStep, ABC, metaclass=AutoRegisterMeta):\n    __registry_key__ = 'step_id'\n    __skip_if_no_key__ = True\n    step_id: ClassVar[str | None] = None\n    registration_order: ClassVar[int] = 0\n\n@dataclass(frozen=True)\nclass {normal_form_class}Matcher:\n    steps: tuple[{normal_form_class}Step, ...]\n\n    def match(self, source):\n        return Maybe.of(source).bind_all(self.steps)"
class EffectStepAmortizationDetector(ConfiguredModuleCollectorCandidateDetector[EffectStepAmortizationCandidate]):
    finding_spec = finding_spec_template(PatternId.STAGED_ORCHESTRATION, 'Manual AST matcher should amortize EffectStep infrastructure', 'A helper that repeatedly performs AST type/cardinality checks and exits through `return None` is paying the cognitive cost of an effect pipeline without reusing the nominal `EffectStep` carrier. The infrastructure pays rent when these guard atoms become registered matcher-step objects that can be shared, ordered, tested, and composed.', 'reusable nominal EffectStep family for recurring AST type, cardinality, and optional-exit guards', 'same optional AST matcher mechanics are hand-expanded inside one helper', _SHARED_ALGORITHM_AUTHORITY_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS, _PREDICATE_CHAIN_NORMALIZED_AST_DATAFLOW_ROOT_OBSERVATION_TAGS)
    def _finding_for_candidate(self, payoff_candidate: EffectStepAmortizationCandidate) -> RefactorFinding:
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
                f"normal form is `{payoff_candidate.normal_form}`."
            ),
            (payoff_candidate.evidence,),
            scaffold=_effect_step_payoff_scaffold(payoff_candidate),
            codemod_patch=(
                '# Extract the repeated AST guard atoms into nominal `EffectStep` subclasses.\n# Register ordered steps with `AutoRegisterMeta`, then route the helper through `Maybe.of(source).bind_all(steps)`.\n# Keep only domain-specific witness construction outside the shared matcher pipeline.'
            ),
            metrics=OrchestrationMetrics(
                function_line_count=payoff_candidate.line_count,
                branch_site_count=payoff_candidate.none_return_count,
                call_site_count=payoff_candidate.semantic_helper_count,
                parameter_count=len(payoff_candidate.ast_type_names),
                callee_family_count=max(
                    1, len(payoff_candidate.semantic_helper_names)
                ),
            ),
        )
declare_module_detector(EffectStepImplementationLeakCandidate, high_confidence_spec(PatternId.ABC_TEMPLATE_METHOD, 'EffectStep leaf should declare hooks instead of owning apply', 'Concrete effect-step leaves should carry semantic residue as attributes/properties and small hooks. When a leaf owns raw optional exits, AST type checks, or cardinality checks inside `apply()` or a bulky hook, the ABC is not doing enough of the work and the monadic infrastructure is not compressing semantics.', 'template-method EffectStep base that owns optional flow, type narrowing, and guard sequencing', 'concrete EffectStep leaf repeats mechanics that belong in an ABC/template base', _FAIL_LOUD_CONTRACTS_SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_CAPABILITY_TAGS, _PREDICATE_CHAIN_NORMALIZED_AST_OBSERVATION_TAGS), CandidateFindingRenderer[EffectStepImplementationLeakCandidate](summary=lambda leak: f'`{leak.class_name}.{leak.method_name}` owns {leak.raw_guard_count} raw guard mechanics and {leak.none_return_count} optional exits; move the algorithm into `{leak.suggested_base_name}` and leave only attrs/properties plus hooks on the leaf.', evidence=lambda leak: (leak.evidence,), scaffold=lambda leak: f"class {leak.class_name}({leak.suggested_base_name}):\n    step_id = 'semantic_step'\n    registration_order = 10\n    # declare class attrs/properties here\n\n    def accepts(self, value): ...\n    def project(self, value): ...", codemod_patch=lambda leak: '# Delete the concrete mechanics-heavy leaf method.\n# Move optional flow/type narrowing/cardinality mechanics to the ABC/template base.\n# Keep the implementation class declarative: attrs, properties, and the smallest semantic hooks.', metrics=lambda leak: OrchestrationMetrics(function_line_count=0, branch_site_count=leak.none_return_count, call_site_count=leak.raw_guard_count, parameter_count=0, callee_family_count=1)), candidate_collector=_effect_step_implementation_leak_candidates)
class UnderAmortizedInfrastructureDetector(CrossModuleCollectorCandidateDetector[UnderAmortizedInfrastructureCandidate]):
    finding_spec = finding_spec_template(PatternId.STAGED_ORCHESTRATION, 'Matcher infrastructure should pay rent through fanout', 'A shared matcher/effect infrastructure module should earn its declarations through repeated external use. When a public helper or carrier has only one external consumer and is not support for a broadly reused declaration, the abstraction is expanding the surface area faster than it compresses manual code.', 'public matcher infrastructure whose declaration cost is amortized by multiple consumers', 'effect/matcher module public surface has single-consumer declarations', _SHARED_ALGORITHM_AUTHORITY_UNIT_RATE_COHERENCE_NOMINAL_IDENTITY_CAPABILITY_TAGS, _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS)
    def _finding_for_candidate(self, under_amortized: UnderAmortizedInfrastructureCandidate) -> RefactorFinding:
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
                '# Either inline the single-consumer declaration into its only consumer, or merge it into an already-amortized primitive.\n# Keep new public matcher infrastructure only when fanout shows more than one external consumer.'
            ),
            codemod_patch=(
                '# Collapse the single-consumer public surface before adding more matcher machinery.\n# If the declaration represents real reusable semantics, route at least two consumers through it.'
            ),
            metrics=OrchestrationMetrics(
                function_line_count=0,
                branch_site_count=len(under_amortized.declaration_names),
                call_site_count=len(under_amortized.consumer_symbols),
                parameter_count=len(under_amortized.support_names),
                callee_family_count=1,
            ),
        )
declare_module_detector(CandidateCollectorBoilerplateCandidate, high_confidence_spec(PatternId.STAGED_ORCHESTRATION, 'Candidate detector should declare collector strategy', 'Detector classes repeatedly implement `_candidate_items()` as a one-line forwarding method. That is boilerplate control flow: the detector identity and finding rendering are semantic, while candidate collection is a typed class-level strategy that can be inherited.', 'typed metaprogrammed detector base that derives candidate collection from a declared strategy', 'detector class repeats collector forwarding method instead of declaring a collector', _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_UNIT_RATE_COHERENCE_CAPABILITY_TAGS, _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS), CandidateFindingRenderer[CandidateCollectorBoilerplateCandidate](summary=lambda collector: f'`{collector.class_name}.{collector.method_name}` only forwards to `{collector.collector_name}`; inherit `{collector.recommended_base_name}` and declare `candidate_collector` instead.', evidence=lambda collector: (collector.evidence,), scaffold=lambda collector: f'class {collector.class_name}({collector.recommended_base_name}):\n    candidate_collector = {collector.collector_name}\n', codemod_patch=lambda collector: f'# Delete the forwarding `_candidate_items()` method.\n# Change the detector base to `{collector.recommended_base_name}` and assign `candidate_collector = {collector.collector_name}`.', metrics=lambda collector: OrchestrationMetrics(function_line_count=0, branch_site_count=1, call_site_count=1, parameter_count=2 if collector.uses_config else 1, callee_family_count=1)), detector_priority=-19, candidate_collector=_candidate_collector_boilerplate_candidates)
declare_module_detector(TypedCandidateCastBoilerplateCandidate, high_confidence_spec(PatternId.ABC_TEMPLATE_METHOD, 'Candidate template method should receive typed candidates directly', 'Detector classes repeatedly accept `candidate: object`, immediately cast it to a nominal candidate type, and then never use the object-typed parameter again. That cast belongs in the generic detector base contract: the implementation hook should receive the typed candidate directly.', 'generic typed candidate detector base with no per-detector cast prelude', 'candidate-rendering template method starts with a local cast of its only payload parameter', _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_CAPABILITY_TAGS, _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS), CandidateFindingRenderer[TypedCandidateCastBoilerplateCandidate](summary=lambda candidate: f'`{candidate.class_name}.{candidate.method_name}` casts `{candidate.parameter_name}` to `{candidate.candidate_type_name}` before doing real work; parameterize `{candidate.detector_base_name}` and receive `{candidate.local_name}` as that type.', evidence=lambda candidate: (candidate.evidence,), scaffold=lambda candidate: f'class {candidate.class_name}({candidate.detector_base_name}[{candidate.candidate_type_name}]):\n    def {candidate.method_name}(self, {candidate.local_name}: {candidate.candidate_type_name}) -> RefactorFinding:\n        ...', codemod_patch=lambda candidate: f'# Change the detector base to `{candidate.detector_base_name}[{candidate.candidate_type_name}]`.\n# Rename the hook argument from `{candidate.parameter_name}` to `{candidate.local_name}` and delete the local `cast(...)` prelude.', metrics=lambda candidate: _SINGLE_TEMPLATE_CALL_METRICS), detector_priority=-18, candidate_collector=_typed_candidate_cast_boilerplate_candidates)
declare_module_detector(DeclarativeDetectorClassCandidate, high_confidence_certified_spec(PatternId.AUTHORITATIVE_SCHEMA, 'Metadata-only detector class should be declared through detector algebra', 'A detector class whose body only assigns finding metadata and a renderer is not carrying implementation behavior. Its class shell is derivable from the candidate type, detector base, registry key, and declaration line.', 'one detector-declaration algebra that derives metadata-only detector classes', 'detector class repeats a nominal class shell around only declarative assignments', _AUTHORITATIVE_NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS, _CLASS_FAMILY_NORMALIZED_AST_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS), CandidateFindingRenderer[DeclarativeDetectorClassCandidate](summary=lambda candidate: f'`{candidate.class_name}` is a {candidate.line_count}-line metadata-only detector over `{candidate.candidate_type_name}` with assignments {candidate.assignment_names}.', evidence=lambda candidate: (candidate.evidence,), scaffold=lambda candidate: f'declare_module_detector({candidate.candidate_type_name}, finding_spec, finding_renderer, detector_base={candidate.base_name})', codemod_patch=lambda candidate: f'# Replace `{candidate.class_name}` with `declare_module_detector(...)`.\n# Keep only true detector-specific values: spec, renderer, optional collector, base, and priority.', metrics=lambda candidate: MappingMetrics.from_field_names(mapping_site_count=candidate.line_count, mapping_name=candidate.class_name, field_names=candidate.assignment_names, source_name=candidate.base_name)), detector_priority=-17, candidate_collector=_declarative_detector_class_candidates)
class FindingSpecDefaultFieldBoilerplateDetector(ModuleCollectorCandidateDetector[FindingSpecDefaultFieldCandidate]):
    candidate_collector = _finding_spec_default_field_candidates
    finding_spec = high_confidence_spec(PatternId.AUTHORITATIVE_SCHEMA, 'FindingSpec semantic defaults should be constructor-derived', 'FindingSpec constructors already encode confidence and certification defaults. Restating those semantic fields in every detector is declaration boilerplate; the constructor should carry the shared semantic tier and leave only true local residue.', 'constructor-level semantic spec defaults with no repeated confidence/certification payload', 'FindingSpec call repeats semantic default keywords that can be derived from its constructor', _AUTHORITATIVE_SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_CAPABILITY_TAGS, _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS)
    def _finding_for_candidate(self, field_candidate: FindingSpecDefaultFieldCandidate) -> RefactorFinding:
        keyword_summary = ', '.join((f'{name}={value}' for name, value in zip(field_candidate.redundant_keyword_names, field_candidate.redundant_keyword_values, strict=True)))
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
                f'{field_candidate.recommended_constructor_name}(\n    pattern_id=...,\n    title=...,\n    ...\n)'
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
declare_module_detector(ClassMethodLineWitnessCandidate, high_confidence_spec(PatternId.ABC_TEMPLATE_METHOD, 'Detector finding builder should derive detector_id', 'Concrete detectors repeatedly call `self.finding_spec.build(self.detector_id, ...)`. The detector id is instance-owned template context, not per-finding payload; a shared `build_finding(...)` hook should inject it once.', 'typed detector template method that injects detector identity into finding construction', 'finding renderer manually passes detector-owned identity into its own spec builder', _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_AUTHORITATIVE_CAPABILITY_TAGS, _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS), CandidateFindingRenderer[ClassMethodLineWitnessCandidate](summary=lambda candidate: f'`{candidate.symbol}` calls `self.finding_spec.build(self.detector_id, ...)`; `build_finding(...)` can derive the detector id from the instance.', evidence=lambda candidate: (candidate.evidence,), scaffold=lambda candidate: 'return self.build_finding(\n    summary,\n    evidence,\n    ...\n)', codemod_patch=lambda candidate: '# Replace `self.finding_spec.build(` with `self.build_finding(`.\n# Delete the leading `self.detector_id,` argument.', metrics=lambda candidate: _SINGLE_TEMPLATE_CALL_METRICS), detector_name='FindingSpecBuildBoilerplateDetector', candidate_collector=_finding_spec_build_boilerplate_candidates)
class DirectBuildFindingRendererDetector(ModuleCollectorCandidateDetector[DirectBuildFindingRendererCandidate]):
    finding_spec = high_confidence_spec(PatternId.ABC_TEMPLATE_METHOD, 'Direct build_finding renderer should be a typed renderer value', 'A `_finding_for_candidate` method whose entire body is `return self.build_finding(...)` does not own control flow. It is a data renderer over one candidate type, so the candidate-to-finding algorithm should live once in the ABC and the detector should supply a typed renderer object.', 'typed candidate finding renderer reused by detector ABC machinery', 'detector method is only a build_finding payload declaration', _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS, _METHOD_ROLE_NORMALIZED_AST_PARTIAL_VIEW_OBSERVATION_TAGS)
    def _finding_for_candidate(self, renderer: DirectBuildFindingRendererCandidate) -> RefactorFinding:
        keyword_summary = ", ".join(renderer.keyword_names) or "no keywords"
        return self.build_finding(
            (
                f"`{renderer.class_name}.{renderer.method_name}` is a direct "
                f"`build_finding(...)` renderer with {renderer.positional_arg_count} "
                f"positional payloads and {keyword_summary}."
            ),
            (renderer.evidence,),
            scaffold=(
                'finding_renderer = CandidateFindingRenderer[Candidate](\n    summary=lambda candidate: ...,\n    evidence=lambda candidate: ...,\n)'
            ),
            codemod_patch=(
                f'# Move the `{renderer.method_name}` payload in `{renderer.class_name}` to a `CandidateFindingRenderer` classvar.\n# Let `CandidateFindingDetector._finding_for_candidate` run the renderer.'
            ),
            metrics=MappingMetrics.from_field_names(
                mapping_site_count=1,
                mapping_name=renderer.class_name,
                field_names=('summary', 'evidence', *renderer.keyword_names),
            ),
        )
declare_module_detector(DerivableDetectorIdCandidate, high_confidence_spec(PatternId.AUTO_REGISTER_META, 'Detector id should derive from detector class name', 'A detector class whose explicit `detector_id` is the snake_case projection of its class name is restating identity already available to the metaclass registry. `AutoRegisterMeta` should derive that key through the detector base.', 'metaclass-derived detector registry key', 'detector class repeats its own name as a manual registry key', _CLASS_LEVEL_REGISTRATION_NOMINAL_IDENTITY_ENUMERATION_CAPABILITY_TAGS, _CLASS_FAMILY_REGISTRY_POPULATION_REPEATED_METHOD_ROLES_OBSERVATION_TAGS), CandidateFindingRenderer[DerivableDetectorIdCandidate](summary=lambda candidate: f'`{candidate.class_name}` declares `detector_id = "{candidate.detector_id_value}"`, which is derivable from the class name.', evidence=lambda candidate: (candidate.evidence,), scaffold=lambda candidate: 'class IssueDetector(ABC, metaclass=AutoRegisterMeta):\n    __key_extractor__ = staticmethod(_detector_id_from_class_name)', codemod_patch=lambda candidate: f'# Delete `detector_id = "{candidate.detector_id_value}"` from `{candidate.class_name}` and let the metaclass derive it.', metrics=lambda candidate: RegistrationMetrics.from_class_names(registration_site_count=1, registry_name='IssueDetector', class_names=(candidate.class_name,))), candidate_collector=_derivable_detector_id_candidates)
declare_module_detector(DerivableCandidateCollectorCandidate, high_confidence_spec(PatternId.ABC_TEMPLATE_METHOD, 'Candidate collector should derive from detector class name', 'A detector whose collector hook is the snake_case projection of its own class name is declaring a mechanical convention. The collector ABC can derive that hook at class creation and leave only non-standard collector aliases explicit.', 'class-name-derived candidate collector hook', 'detector class repeats its candidate collector naming convention', _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS, _CLASS_FAMILY_REGISTRY_POPULATION_REPEATED_METHOD_ROLES_OBSERVATION_TAGS), CandidateFindingRenderer[DerivableCandidateCollectorCandidate](summary=lambda candidate: f'`{candidate.class_name}` declares `candidate_collector = {candidate.collector_name}`, which is derivable from the class name.', evidence=lambda candidate: (candidate.evidence,), scaffold=lambda candidate: 'class ModuleCollectorCandidateDetector(DerivedCandidateCollectorMixin, ...):\n    ...', codemod_patch=lambda candidate: f'# Delete `candidate_collector = {candidate.collector_name}` from `{candidate.class_name}` and let the collector ABC derive it.', metrics=lambda candidate: MappingMetrics.from_field_names(mapping_site_count=1, mapping_name=candidate.class_name, field_names=('candidate_collector',))), candidate_collector=_derivable_candidate_collector_candidates)
declare_module_detector(CanonicalFindingSpecBuilderCandidate, high_confidence_spec(PatternId.AUTHORITATIVE_SCHEMA, 'FindingSpec coordinates should use one typed semantic builder', 'Detector specs repeatedly enumerate the same semantic coordinate names: pattern, title, why, capability gap, relation context, and tag axes. A typed builder can make that product structure explicit once and leave each detector to provide only its coordinate values.', 'typed FindingSpec builder with canonical semantic coordinate order', 'detector repeats FindingSpec keyword schema locally', _AUTHORITATIVE_SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_CAPABILITY_TAGS, _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS), CandidateFindingRenderer[CanonicalFindingSpecBuilderCandidate](summary=lambda candidate: f'`{candidate.class_name}` builds `{candidate.constructor_name}` by spelling {len(candidate.keyword_names)} FindingSpec coordinate keywords; use `{candidate.builder_name}`.', evidence=lambda candidate: (candidate.evidence,), scaffold=lambda candidate: f'finding_spec = {candidate.builder_name}(\n    PatternId.EXAMPLE,\n    title,\n    why,\n    capability_gap,\n    relation_context,\n)', codemod_patch=lambda candidate: f'# Replace `{candidate.constructor_name}(pattern_id=..., title=..., ...)` with `{candidate.builder_name}(...)` and let the builder own coordinate names.', metrics=lambda candidate: MappingMetrics.from_field_names(mapping_site_count=1, mapping_name=candidate.class_name, field_names=candidate.keyword_names)), candidate_collector=_canonical_finding_spec_builder_candidates)
declare_module_detector(ManualSortedTupleReturnCandidate, high_confidence_certified_spec(PatternId.LOCAL_VALUE_AUTHORITY, 'Manual sorted tuple finalization should use collection algebra', 'A return value shaped as `tuple(sorted(...))` is a standard immutable ordering projection. Spelling the constructor nesting at each site repeats collection mechanics instead of naming the algebraic operation once.', 'typed sorted_tuple collection algebra reused across finalization sites', 'function manually nests tuple construction around sorted ordering', _SHARED_ALGORITHM_AUTHORITY_AUTHORITATIVE_NOMINAL_IDENTITY_CAPABILITY_TAGS, _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS), CandidateFindingRenderer[ManualSortedTupleReturnCandidate](summary=lambda candidate: f'`{candidate.qualname}` returns a manual `tuple(sorted(...))` finalizer over `{candidate.sorted_expression}` spanning {candidate.line_count} line(s).', evidence=lambda candidate: (candidate.evidence,), scaffold=lambda candidate: 'from nominal_refactor_advisor.collection_algebra import sorted_tuple\n\nreturn sorted_tuple(items, key=key_function)', codemod_patch=lambda candidate: '# Replace `return tuple(sorted(...))` with `return sorted_tuple(...)` so tuple immutability and ordering are one named collection algebra.', metrics=lambda candidate: MappingMetrics.from_field_names(mapping_site_count=1, mapping_name=candidate.qualname, field_names=tuple((name for name, value in (('items', candidate.sorted_expression), ('key', candidate.key_expression), ('reverse', candidate.reverse_expression)) if value is not None)))), candidate_collector=_manual_sorted_tuple_return_candidates)
declare_module_detector(ManualSortedTupleExpressionCandidate, high_confidence_certified_spec(PatternId.LOCAL_VALUE_AUTHORITY, 'Manual sorted tuple expression should use collection algebra', 'A nested `tuple(sorted(...))` expression is a standard immutable ordering projection. When it appears inside assignments, constructor payloads, or comprehensions, it is still collection mechanics that should be named once by the shared algebra.', 'typed sorted_tuple collection algebra reused inside expression payloads', 'expression manually nests tuple construction around sorted ordering', _SHARED_ALGORITHM_AUTHORITY_AUTHORITATIVE_NOMINAL_IDENTITY_CAPABILITY_TAGS, _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS), CandidateFindingRenderer[ManualSortedTupleExpressionCandidate](summary=lambda candidate: f'`{candidate.qualname}` contains a manual `tuple(sorted(...))` {candidate.context_kind} expression over `{candidate.sorted_expression}` spanning {candidate.line_count} line(s).', evidence=lambda candidate: (candidate.evidence,), scaffold=lambda candidate: 'from nominal_refactor_advisor.collection_algebra import sorted_tuple\n\nvalue = sorted_tuple(items, key=key_function)', codemod_patch=lambda candidate: '# Replace nested `tuple(sorted(...))` with `sorted_tuple(...)` so expression payloads use the shared collection algebra.', metrics=lambda candidate: MappingMetrics.from_field_names(mapping_site_count=1, mapping_name=candidate.qualname, field_names=tuple((name for name, value in (('items', candidate.sorted_expression), ('key', candidate.key_expression), ('reverse', candidate.reverse_expression)) if value is not None)))), candidate_collector=_manual_sorted_tuple_expression_candidates)
declare_module_detector(SimplePropertyAliasClassCandidate, high_confidence_certified_spec(PatternId.LOCAL_VALUE_AUTHORITY, 'Property alias class should use descriptor algebra', 'A class whose only concrete behavior is returning `self.<field>` from properties is a structural alias shell. The alias relation is the semantic object; repeated property methods re-declare descriptor mechanics instead of naming that relation directly.', 'typed alias-property descriptor derived from declared source and target names', 'class repeats property method bodies for direct field projection', _SHARED_ALGORITHM_AUTHORITY_AUTHORITATIVE_NOMINAL_IDENTITY_CAPABILITY_TAGS, _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS), CandidateFindingRenderer[SimplePropertyAliasClassCandidate](summary=lambda candidate: f'`{candidate.class_name}` defines {len(candidate.alias_pairs)} direct property alias(es) across {candidate.line_count} line(s): ' + ', '.join((f'{target} -> {source}' for target, source in candidate.alias_pairs)), evidence=lambda candidate: (candidate.evidence,), scaffold=lambda candidate: 'from nominal_refactor_advisor.descriptor_algebra import AliasProperty\n\nclass Shape:\n    target = AliasProperty[ValueType]("source")', codemod_patch=lambda candidate: '# Replace direct `@property return self.<source>` alias methods with `AliasProperty[...]` descriptors so alias projection is one typed descriptor algebra.', metrics=lambda candidate: MappingMetrics.from_field_names(mapping_site_count=len(candidate.alias_pairs), mapping_name=candidate.class_name, field_names=tuple((f'{target}->{source}' for target, source in candidate.alias_pairs)))), candidate_collector=_simple_property_alias_class_candidates)
declare_module_detector(SimplePropertyAliasMethodCandidate, high_confidence_certified_spec(PatternId.LOCAL_VALUE_AUTHORITY, 'Direct property alias method should use descriptor algebra', 'A property method whose body is exactly `return self.<field>` is a descriptor relation, even when the surrounding class owns other behavior. Keeping that relation as a method repeats alias mechanics and hides the source-target projection from class declarations.', 'typed alias-property descriptor reused for direct field projection methods', 'property method repeats direct self-field alias mechanics', _SHARED_ALGORITHM_AUTHORITY_AUTHORITATIVE_NOMINAL_IDENTITY_CAPABILITY_TAGS, _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS), CandidateFindingRenderer[SimplePropertyAliasMethodCandidate](summary=lambda candidate: f'`{candidate.class_name}.{candidate.method_name}` is a direct property alias to `{candidate.source_name}`.', evidence=lambda candidate: (candidate.evidence,), scaffold=lambda candidate: f'''from nominal_refactor_advisor.descriptor_algebra import AliasProperty\n\n{candidate.method_name} = AliasProperty[{candidate.return_annotation or 'ValueType'}]("{candidate.source_name}")''', codemod_patch=lambda candidate: '# Replace the `@property return self.<source>` method with an `AliasProperty[...]` descriptor on the class body.', metrics=lambda candidate: MappingMetrics.from_field_names(mapping_site_count=1, mapping_name=f'{candidate.class_name}.{candidate.method_name}', field_names=(candidate.source_name,))), candidate_collector=_simple_property_alias_method_candidates)
declare_module_detector(FieldOnlyFrozenDataclassCandidate, high_confidence_certified_spec(PatternId.AUTHORITATIVE_SCHEMA, 'Field-only frozen dataclass should use product-record algebra', 'A frozen dataclass whose body contains only field annotations is a nominal product. Spelling the decorator, class shell, and one field declaration per line repeats record mechanics that can be derived from a compact product schema.', 'frozen nominal product class derived from one product-record schema', 'field-only frozen dataclass repeats product-record declaration mechanics', _SHARED_ALGORITHM_AUTHORITY_AUTHORITATIVE_NOMINAL_IDENTITY_CAPABILITY_TAGS, _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS), CandidateFindingRenderer[FieldOnlyFrozenDataclassCandidate](summary=lambda candidate: f'`{candidate.class_name}` is a {len(candidate.field_specs)}-field frozen product record spanning {candidate.line_count} line(s).', evidence=lambda candidate: (candidate.evidence,), scaffold=lambda candidate: f'from nominal_refactor_advisor.record_algebra import product_record\n\n{candidate.class_name} = product_record(\n    "{candidate.class_name}",\n    "field_name: FieldType; other_field: OtherType",\n)', codemod_patch=lambda candidate: '# Replace the field-only `@dataclass(frozen=True)` class shell with `product_record(...)`, preserving bases, field annotations, defaults, docstring, and dataclass keyword-only semantics.', metrics=lambda candidate: MappingMetrics.from_field_names(mapping_site_count=1, mapping_name=candidate.class_name, field_names=tuple((name for name, _ in candidate.field_specs)))), candidate_collector=_field_only_frozen_dataclass_candidates)
declare_module_detector(DuplicateVisitorMethodBodyCandidate, high_confidence_certified_spec(PatternId.ABC_TEMPLATE_METHOD, 'Duplicate AST visitor hooks should share one hook implementation', 'Sibling `visit_*` methods with exactly the same normalized body encode one visitor transition more than once. The shared body should be one hook or an explicit method alias, leaving the node-type distinction in dispatch metadata.', 'one visitor hook implementation reused by equivalent node dispatch entries', 'same normalized visitor hook body is repeated on sibling visit methods', _SHARED_ALGORITHM_AUTHORITY_NOMINAL_IDENTITY_MRO_ORDERING_CAPABILITY_TAGS, _NORMALIZED_AST_CLASS_FAMILY_METHOD_ROLE_OBSERVATION_TAGS), CandidateFindingRenderer[DuplicateVisitorMethodBodyCandidate](summary=lambda candidate: f"`{candidate.class_name}` repeats the same visitor body across {', '.join(candidate.method_names)}.", evidence=lambda candidate: (candidate.evidence,), scaffold=lambda candidate: f'{candidate.method_names[0]}(...)\n{candidate.method_names[1]} = {candidate.method_names[0]}', codemod_patch=lambda candidate: '# Replace duplicate sibling `visit_*` method bodies with one shared implementation or explicit aliases for equivalent visitor dispatch entries.', metrics=lambda candidate: RepeatedMethodMetrics.from_duplicate_family(duplicate_site_count=len(candidate.method_names), statement_count=candidate.statement_count, class_count=1, method_symbols=tuple((f'{candidate.class_name}.{method_name}' for method_name in candidate.method_names)))), candidate_collector=_duplicate_visitor_method_body_candidates)
declare_module_detector(EnumMetadataTableCandidate, high_confidence_certified_spec(PatternId.AUTHORITATIVE_SCHEMA, 'Enum metadata table should be carried by enum members', 'An enum whose properties only index a module-level table by `self` splits member identity from member metadata. The metadata should move into enum construction so each member carries its own typed semantic record.', 'enum member construction owns the member metadata', 'enum properties read a parallel metadata table keyed by the same enum family', _AUTHORITATIVE_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS, _EXPORT_MAPPING_NORMALIZED_AST_OBSERVATION_TAGS), CandidateFindingRenderer[EnumMetadataTableCandidate](summary=lambda candidate: f'`{candidate.class_name}` reads {candidate.property_names} from `{candidate.table_name}` across {candidate.case_count} enum cases.', evidence=lambda candidate: (candidate.evidence,), scaffold=lambda candidate: 'class MetadataEnum(StrEnum):\n    def __new__(cls, value: str, label: str):\n        obj = str.__new__(cls, value)\n        obj._value_ = value\n        obj.label = label\n        return obj', codemod_patch=lambda candidate: f'# Move `{candidate.table_name}` values into `{candidate.class_name}` member tuples and delete the table-backed property lookups.', metrics=lambda candidate: MappingMetrics.from_field_names(mapping_site_count=candidate.case_count, mapping_name=candidate.table_name, field_names=candidate.property_names, source_name=candidate.class_name)), candidate_collector=_enum_metadata_table_candidates)
class SemanticTagTupleBoilerplateDetector(ModuleCollectorCandidateDetector[SemanticTagTupleBoilerplateCandidate]):
    finding_spec = high_confidence_spec(PatternId.AUTHORITATIVE_SCHEMA, 'Semantic tag tuple literal should become a named authority', 'Capability and observation tag tuples are semantic classifications, not local control flow. A named constant should carry that classification so detector specs reference one authority instead of re-declaring enum tuples inline.', 'named semantic tag tuple authority reused across detector specs', 'FindingSpec carries an inline semantic tag tuple literal', _AUTHORITATIVE_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS, _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS)
    def _finding_for_candidate(self, tag_candidate: SemanticTagTupleBoilerplateCandidate) -> RefactorFinding:
        if tag_candidate.source_kind == "derived_constant":
            sample_names = ", ".join(tag_candidate.tag_names[:4])
            suffix = f", and {len(tag_candidate.tag_names) - 4} more" if len(tag_candidate.tag_names) > 4 else ""
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
                f'{tag_candidate.constant_name} = (\n    ...\n)\n\nfinding_spec = HighConfidenceFindingSpec({tag_candidate.keyword_name}={tag_candidate.constant_name})'
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
class DerivedMetricCountBoilerplateDetector(ModuleCollectorCandidateDetector[DerivedMetricCountBoilerplateCandidate]):
    finding_spec = high_confidence_spec(PatternId.AUTHORITATIVE_SCHEMA, 'Metric counts should be derived from metric collections', 'A metrics object that receives both `*_count=len(values)` and `values=values` is carrying the same fact twice. The count is a deterministic projection of the collection and should be derived by the typed metrics constructor.', 'typed metrics constructors that derive count fields from authoritative collections', 'metrics call passes a count keyword computed from the collection keyword in the same call', _AUTHORITATIVE_NOMINAL_IDENTITY_PROVENANCE_CAPABILITY_TAGS, _DATAFLOW_ROOT_NORMALIZED_AST_OBSERVATION_TAGS)
    def _finding_for_candidate(self, metric_candidate: DerivedMetricCountBoilerplateCandidate) -> RefactorFinding:
        derived_summary = ', '.join((f'{count_name}=len({collection_name})' for count_name, collection_name in zip(metric_candidate.count_keyword_names, metric_candidate.collection_keyword_names, strict=True)))
        return self.build_finding(
            (
                f"`{metric_candidate.metric_class_name}` repeats derived count fields "
                f"{derived_summary}; use `{metric_candidate.recommended_constructor_name}`."
            ),
            (metric_candidate.evidence,),
            scaffold=(
                f'{metric_candidate.metric_class_name}.{metric_candidate.recommended_constructor_name}(\n    ...\n)'
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
declare_module_detector(DataclassNamespaceCliMirrorCandidate, high_confidence_certified_spec(PatternId.AUTHORITATIVE_SCHEMA, 'Dataclass config surfaces should derive namespace and CLI adapters', 'A dataclass already owns its field names and defaults. Re-enumerating those fields in a namespace constructor and an argparse specification table creates parallel configuration surfaces that can drift from the typed record.', 'one dataclass field authority that derives namespace construction and CLI argument rows', 'dataclass fields are mirrored manually in both from-namespace construction and CLI argument specs', _AUTHORITATIVE_NOMINAL_IDENTITY_SHARED_ALGORITHM_AUTHORITY_CAPABILITY_TAGS, _DATAFLOW_ROOT_NORMALIZED_AST_MANUAL_SYNCHRONIZATION_OBSERVATION_TAGS), CandidateFindingRenderer[DataclassNamespaceCliMirrorCandidate](summary=lambda candidate: f'`{candidate.class_name}` mirrors {len(candidate.field_names)} namespace fields and {len(candidate.cli_field_names)} CLI fields through `{candidate.argument_spec_name}` instead of deriving adapters from the dataclass.', evidence=lambda candidate: (SourceLocation(candidate.file_path, candidate.line, candidate.class_name), SourceLocation(candidate.file_path, candidate.from_namespace_line, f'{candidate.class_name}.from_namespace'), SourceLocation(candidate.argument_spec_file_path, candidate.argument_spec_line, candidate.argument_spec_name)), scaffold=lambda candidate: 'for field in fields(ConfigRecord):\n    value = namespace_values.get(field.name, field.default)\n    ...\n\nCLI_SPECS = tuple(spec_from_field(field) for field in fields(ConfigRecord) if field.name in HELP)', codemod_patch=lambda candidate: f'# Derive `{candidate.class_name}.from_namespace()` and `{candidate.argument_spec_name}` from dataclass fields/defaults.\n# Keep per-option help text as the only CLI-specific residue.', metrics=lambda candidate: MappingMetrics.from_field_names(mapping_site_count=len(candidate.field_names) + len(candidate.cli_field_names), mapping_name=candidate.class_name, field_names=candidate.field_names, source_name=candidate.argument_spec_name)), detector_base=CrossModuleCollectorCandidateDetector, candidate_collector=_dataclass_namespace_cli_mirror_candidates)
class NestedBuilderShellDetector(ConfiguredModuleCollectorCandidateDetector[NestedBuilderShellCandidate]):
    finding_spec = high_confidence_spec(PatternId.AUTHORITATIVE_CONTEXT, 'Nested builder shell should collapse into one authoritative request boundary', 'A builder forwards a substantial semantic parameter family unchanged into a subordinate nominal builder and only adds a small residue locally. The docs treat that as split request authority: one layer should own the forwarded family instead of rebuilding it inside another shell.', 'single authoritative request/context builder boundary', 'one builder nests a forwarded subordinate request builder inside a second nominal shell', _AUTHORITATIVE_PROVENANCE_UNIT_RATE_COHERENCE_CAPABILITY_TAGS)
    def _finding_for_candidate(self, shell_candidate: NestedBuilderShellCandidate) -> RefactorFinding:
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
                '@dataclass(frozen=True)\nclass OuterRequest:\n    child_request: ChildRequest\n\n    @classmethod\n    def from_source(cls, source, *, child_request: ChildRequest):\n        return cls(child_request=child_request, ...)\n'
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
declare_module_detector(ManualFiberTagCandidate, high_confidence_spec(PatternId.NOMINAL_BOUNDARY, 'Manual fiber tag should become nominal family', 'A string-valued instance tag is manually selecting behavior while the same instance still carries fields from several incompatible fibers. That leaves the family above the zero-incoherence threshold and admits disagreement states the host type system could rule out.', 'host-native nominal fiber decomposition with one subclass per behavior fiber', 'manual instance tag drives behavior while irrelevant coordinates remain constructible on every fiber', _NOMINAL_IDENTITY_PROVENANCE_FAIL_LOUD_CONTRACTS_CAPABILITY_TAGS), CandidateFindingRenderer[ManualFiberTagCandidate](summary=lambda fiber_candidate: f'`{fiber_candidate.class_name}` branches on manual fiber tag `self.{fiber_candidate.tag_name}` across {fiber_candidate.case_names} while still carrying cross-fiber fields {fiber_candidate.assigned_field_names}.', evidence=lambda fiber_candidate: (SourceLocation(fiber_candidate.file_path, fiber_candidate.init_line, f'{fiber_candidate.class_name}.__init__'), SourceLocation(fiber_candidate.file_path, fiber_candidate.method_line, f'{fiber_candidate.class_name}.{fiber_candidate.method_name}')), scaffold=lambda fiber_candidate: _manual_fiber_tag_scaffold(fiber_candidate), codemod_patch=lambda fiber_candidate: _manual_fiber_tag_patch(fiber_candidate), metrics=lambda fiber_candidate: DispatchCountMetrics.from_literal_family(dispatch_axis=f'self.{fiber_candidate.tag_name}', literal_cases=fiber_candidate.case_names)), candidate_collector=_manual_fiber_tag_candidates)
declare_module_detector(DescriptorDerivedViewCandidate, high_confidence_spec(PatternId.DESCRIPTOR_DERIVED_VIEW, 'Derived views stored independently of their source', 'Several stored fields are derived from one authoritative source field, but mutators resynchronize them manually and incompletely. That raises the degree of freedom above one and makes view disagreement reachable.', 'descriptor- or property-mediated derived views rooted in one authoritative source', 'stored derived views must be manually kept coherent with a single source field', _AUTHORITATIVE_UNIT_RATE_COHERENCE_PROVENANCE_CAPABILITY_TAGS), CandidateFindingRenderer[DescriptorDerivedViewCandidate](summary=lambda view_candidate: f'`{view_candidate.class_name}` stores derived views {view_candidate.derived_field_names} from `{view_candidate.source_attr}`, but `{view_candidate.mutator_name}` only updates {view_candidate.updated_field_names}.', evidence=lambda view_candidate: (SourceLocation(view_candidate.file_path, view_candidate.init_line, f'{view_candidate.class_name}.__init__'), SourceLocation(view_candidate.file_path, view_candidate.mutator_line, f'{view_candidate.class_name}.{view_candidate.mutator_name}')), scaffold=lambda view_candidate: _descriptor_derived_view_scaffold(view_candidate), codemod_patch=lambda view_candidate: _descriptor_derived_view_patch(view_candidate)), candidate_collector=_descriptor_derived_view_candidates)
class DeferredClassRegistrationDetector(ModuleCollectorCandidateDetector[ManualRegistryCandidate]):
    candidate_collector = _manual_registry_candidates
    finding_spec = high_confidence_spec(PatternId.AUTO_REGISTER_META, 'Class registration is decoupled from class existence', 'Manual decorator- or helper-based registration leaves a reachable state where a class exists but the registry has not been updated. The host already provides zero-delay registration via `metaclass-registry` or another class-time hook.', 'zero-delay metaclass-registry class registration with collision checks and runtime provenance', 'class registration is performed as a separate auxiliary step rather than at class creation time', _CLASS_LEVEL_REGISTRATION_PROVENANCE_NOMINAL_IDENTITY_CAPABILITY_TAGS)
    def _finding_for_candidate(self, registry_candidate: ManualRegistryCandidate) -> RefactorFinding: evidence = [SourceLocation(registry_candidate.file_path, registry_candidate.line, registry_candidate.decorator_name)]; evidence.extend((SourceLocation(registry_candidate.file_path, registry_candidate.line, class_name) for class_name in registry_candidate.class_names[:5])); return self.build_finding(f'Registry `{registry_candidate.registry_name}` is updated through manual decorator `{registry_candidate.decorator_name}` for classes {registry_candidate.class_names}, leaving registration structurally decoupled from class creation.', tuple(evidence), scaffold=_manual_registry_scaffold(registry_candidate), codemod_patch=_manual_registry_patch(registry_candidate), metrics=RegistrationMetrics(registration_site_count=len(registry_candidate.class_names), registry_name=registry_candidate.registry_name))
class StructuralConfusabilityDetector(ModuleCollectorCandidateDetector[StructuralConfusabilityCandidate]):
    finding_spec = high_confidence_spec(PatternId.NOMINAL_INTERFACE_WITNESS, 'Consumer observes a confusable duck-typed family', 'A consumer only observes a partial structural view, and several unrelated classes are confusable under that view. Without a nominal witness, the distortion floor stays above zero and the family boundary remains implicit.', 'ABC-backed nominal witness for a structurally confusable implementation family', 'consumer depends on a partial structural view shared by several unrelated classes', _NOMINAL_IDENTITY_FAIL_LOUD_CONTRACTS_PROVENANCE_CAPABILITY_TAGS)
    def _finding_for_candidate(self, confusability_candidate: StructuralConfusabilityCandidate) -> RefactorFinding: evidence = (SourceLocation(confusability_candidate.file_path, confusability_candidate.line, confusability_candidate.function_name),); return self.build_finding(f'`{confusability_candidate.function_name}` observes `{confusability_candidate.parameter_name}` only through methods {confusability_candidate.observed_method_names}, but classes {confusability_candidate.class_names} are confusable under that view.', evidence, scaffold=_structural_confusability_scaffold(confusability_candidate), codemod_patch=_structural_confusability_patch(confusability_candidate))
class SemanticWitnessFamilyDetector(ModuleCollectorCandidateDetector[WitnessCarrierFamilyCandidate]):
    candidate_collector = _witness_carrier_family_candidates
    finding_spec = high_confidence_spec(PatternId.NOMINAL_WITNESS_CARRIER, 'Semantic carrier family should share one nominal base', 'Several frozen dataclass carriers repeat the same location and naming roles under different field names. That leaves one semantic family structurally expanded instead of giving it one nominal carrier root.', 'one authoritative nominal base for a semantic metadata carrier family', 'same carrier family repeats a renamed semantic-role spine across sibling frozen dataclasses', _NOMINAL_IDENTITY_PROVENANCE_AUTHORITATIVE_CAPABILITY_TAGS)
    def _finding_for_candidate(self, witness_candidate: WitnessCarrierFamilyCandidate) -> RefactorFinding: evidence = tuple((SourceLocation(witness_candidate.file_path, line, class_name) for class_name, line in zip(witness_candidate.class_names, witness_candidate.line_numbers, strict=True))); return self.build_finding(f"Frozen carrier classes {', '.join(witness_candidate.class_names)} repeat semantic roles {witness_candidate.shared_role_names} under renamed fields and should inherit one nominal base carrier.", evidence, scaffold=_witness_carrier_family_scaffold(witness_candidate), codemod_patch=_witness_carrier_family_patch(witness_candidate), metrics=WitnessCarrierMetrics(class_count=len(witness_candidate.class_names), shared_role_count=len(witness_candidate.shared_role_names), class_names=witness_candidate.class_names, shared_role_names=witness_candidate.shared_role_names))
