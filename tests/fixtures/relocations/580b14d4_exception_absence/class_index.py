# Extracted from 580b14d4:nominal_refactor_advisor/class_index.py


@dataclass(frozen=True)
class LatentRosterObservation:
    """One source roster, with parsing and family matching owned once."""

    policy_tokens: ClassVar[frozenset[str]] = frozenset(
        {
            "active",
            "all",
            "available",
            "default",
            "enabled",
            "public",
            "selected",
            "supported",
            "test",
            "visible",
        }
    )
    policy_noise_tokens: ClassVar[frozenset[str]] = frozenset(
        {"by", "classes", "formats", "map", "registry", "types"}
    )
    mutation_methods: ClassVar[frozenset[str]] = frozenset({"extend", "update"})

    file_path: str
    roster_name: str
    line: int
    roster_kind: str
    projection_role: str
    member_names: tuple[str, ...]
    line_count: int

    @property
    def is_public_export_surface(self) -> bool:
        """Return whether this roster declares Python's module export contract."""

        return self.roster_name == PYTHON_PUBLIC_EXPORT_ASSIGNMENT

    @classmethod
    def from_module(
        cls,
        parsed_module: ParsedModule,
    ) -> tuple[LatentRosterObservation, ...]:
        observations: list[LatentRosterObservation] = []
        for statement in _trim_leading_docstring(list(parsed_module.module.body)):
            observations.extend(cls.from_statement(parsed_module, statement))
            observations.extend(cls.from_mutation_statement(parsed_module, statement))
            if not isinstance(statement, ast.ClassDef):
                continue
            for class_statement in _trim_leading_docstring(list(statement.body)):
                observations.extend(
                    cls.from_statement(
                        parsed_module,
                        class_statement,
                        roster_prefix=statement.name,
                    )
                )
                observations.extend(
                    cls.from_mutation_statement(parsed_module, class_statement)
                )
        return tuple(observations)

    @classmethod
    def from_statement(
        cls,
        parsed_module: ParsedModule,
        statement: ast.stmt,
        *,
        roster_prefix: str | None = None,
    ) -> tuple[LatentRosterObservation, ...]:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = _trim_leading_docstring(list(statement.body))
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                return ()
            try:
                value = StatementValue.of(body[0]).require_value("a bare return")
            except AbsentStatementValue:
                return ()
            return cls.from_value(
                parsed_module,
                statement,
                roster_name=statement.name,
                value=value,
            )
        target_value: tuple[ast.AST, ast.AST] | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target_value = statement.targets[0], statement.value
        elif isinstance(statement, ast.AnnAssign):
            target_value = StatementValue.of(statement).paired_with(statement.target)
        if target_value is None or not isinstance(target_value[0], ast.Name):
            return ()
        target, value = target_value
        roster_name = (
            f"{roster_prefix}.{target.id}" if roster_prefix is not None else target.id
        )
        return cls.from_value(
            parsed_module,
            statement,
            roster_name=roster_name,
            value=value,
        )

    @classmethod
    def from_mutation_statement(
        cls,
        parsed_module: ParsedModule,
        statement: ast.stmt,
    ) -> tuple[LatentRosterObservation, ...]:
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr in cls.mutation_methods
        ):
            return ()
        call = statement.value
        return tuple(
            observation.for_mutation(call.func.attr)
            for argument in call.args
            for observation in cls.from_value(
                parsed_module,
                statement,
                roster_name=ast.unparse(call.func.value),
                value=argument,
            )
        )

    @classmethod
    def from_value(
        cls,
        parsed_module: ParsedModule,
        statement: ast.stmt,
        *,
        roster_name: str,
        value: ast.AST,
    ) -> tuple[LatentRosterObservation, ...]:
        if isinstance(value, ast.Dict):
            return tuple(
                cls.from_members(
                    parsed_module,
                    statement,
                    roster_name=roster_name,
                    value=value,
                    projection_role=projection_role,
                    member_names=member_names,
                )
                for projection_role, member_names in (
                    ("dict_keys", cls.dict_member_names(value.keys)),
                    ("dict_values", cls.dict_member_names(value.values)),
                )
                if len(member_names) >= 2
            )
        member_names = cls.names_from(value)
        if len(member_names) < 2:
            return ()
        return (
            cls.from_members(
                parsed_module,
                statement,
                roster_name=roster_name,
                value=value,
                projection_role="collection_members",
                member_names=member_names,
            ),
        )

    @classmethod
    def from_members(
        cls,
        parsed_module: ParsedModule,
        statement: ast.stmt,
        *,
        roster_name: str,
        value: ast.AST,
        projection_role: str,
        member_names: tuple[str, ...],
    ) -> LatentRosterObservation:
        return cls(
            file_path=parsed_module.file_path,
            roster_name=roster_name,
            line=statement.lineno,
            roster_kind=type(value).__name__,
            projection_role=projection_role,
            member_names=member_names,
            line_count=(statement.end_lineno or statement.lineno)
            - statement.lineno
            + 1,
        )

    @staticmethod
    def names_from(node: ast.AST) -> tuple[str, ...]:
        if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return ()
        member_names: set[str] = set()
        for element in node.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                member_names.add(element.value)
                continue
            reference = element.func if isinstance(element, ast.Call) else element
            if (member_name := _terminal_reference_name(reference)) is not None:
                member_names.add(member_name)
        return sorted_tuple(member_names)

    @classmethod
    def dict_member_names(cls, nodes: list[ast.expr | None]) -> tuple[str, ...]:
        return sorted_tuple(
            {
                member_name
                for node in nodes
                if node is not None
                for member_name in cls.names_from(
                    ast.Tuple(elts=[node], ctx=ast.Load())
                )
            }
        )

    def for_mutation(self, mutation_name: str) -> LatentRosterObservation:
        return replace(
            self,
            roster_kind=f"inline_{self.roster_kind}.{mutation_name}",
            projection_role=f"{mutation_name}_{self.projection_role}",
        )

    @property
    def projection_policy_hint(self) -> str | None:
        policy_tokens = tuple(
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9]*", self.roster_name)
            if token.lower() not in self.policy_noise_tokens
            and token.lower() in self.policy_tokens
        )
        return "_".join(policy_tokens) or None

    def match(
        self,
        authority_member_names: tuple[str, ...],
    ) -> LatentRosterMatch | None:
        roster_members = set(self.member_names)
        authority_members = set(authority_member_names)
        if not authority_members or not roster_members <= authority_members:
            return None
        missing_names = sorted_tuple(authority_members - roster_members)
        coverage_ratio = len(roster_members) / len(authority_members)
        if not missing_names:
            return LatentRosterMatch(coverage_ratio, (), None)
        if self.projection_policy_hint is None:
            return None
        return LatentRosterMatch(
            coverage_ratio,
            missing_names,
            self.projection_policy_hint,
        )


@dataclass(frozen=True)
class _CompactClassBindingFacets:
    """Class facts whose authority depends on one module binding snapshot."""

    base_references: tuple[CompactNominalReference, ...]
    class_decorators_are_promotion_safe: bool
    dataclass_declaration: CompactDataclassFacet
    enum_declaration: PythonEnumDeclarationFacet

    @property
    def product_base_bindings_are_exact(self) -> bool:
        try:
            for reference in self.base_references:
                if ClassSymbolResolutionAuthority.establishes_nominal_family(
                    ".".join(reference.source_parts)
                ):
                    reference.require_bound_name()
        except UnboundReference:
            return False
        return True


def _dataclass_field_role(
    module_binding_snapshot: ModuleNominalBindingSnapshot,
    class_aliases: dict[str, str],
    statement: ast.AnnAssign,
    preceding_class_bound_names: frozenset[str],
) -> CompactDataclassFieldRole:
    annotation_root = _dataclass_role_reference(statement.annotation)
    qualified_annotation = (
        None
        if annotation_root is None
        else _class_scope_qualified_import_name(
            module_binding_snapshot,
            class_aliases,
            annotation_root,
            preceding_class_bound_names,
        )
    )
    if (
        semantic_role := CompactDataclassFieldRole.for_qualified_annotation(
            qualified_annotation
        )
    ) is not None:
        return semantic_role
    annotation_is_resolved_plain_field = (
        (
            annotation_root is None
            and _dataclass_plain_annotation_is_resolved(
                module_binding_snapshot,
                class_aliases,
                statement.annotation,
                preceding_class_bound_names,
            )
        )
        or qualified_annotation is not None
        or (
            isinstance(annotation_root, ast.Name)
            and module_binding_snapshot.resolves_unshadowed_builtin(
                annotation_root.id,
                preceding_class_bound_names=preceding_class_bound_names,
            )
        )
    )
    if not annotation_is_resolved_plain_field:
        return CompactDataclassFieldRole.UNRESOLVED
    try:
        value = StatementValue.of(statement).require_value("a bare field annotation")
    except AbsentStatementValue:
        return CompactDataclassFieldRole.STORED_INIT
    if isinstance(value, ast.Constant):
        return CompactDataclassFieldRole.STORED_INIT
    if not isinstance(statement.value, ast.Call):
        return CompactDataclassFieldRole.UNRESOLVED
    qualified_default_factory = _class_scope_qualified_import_name(
        module_binding_snapshot,
        class_aliases,
        statement.value.func,
        preceding_class_bound_names,
    )
    default_factory_declaration = DataclassRuntimeDeclaration.for_qualified_name(
        qualified_default_factory
    )
    if (
        default_factory_declaration is None
        or not default_factory_declaration.is_field_factory
    ):
        return CompactDataclassFieldRole.UNRESOLVED
    if statement.value.args or CallKeywords.of(statement.value.keywords).splats:
        return CompactDataclassFieldRole.UNRESOLVED
    init_values = tuple(
        keyword.value for keyword in statement.value.keywords if keyword.arg == "init"
    )
    if len(init_values) > 1:
        return CompactDataclassFieldRole.UNRESOLVED
    if not init_values:
        return CompactDataclassFieldRole.STORED_INIT
    init_value = init_values[0]
    if not isinstance(init_value, ast.Constant) or not isinstance(
        init_value.value,
        bool,
    ):
        return CompactDataclassFieldRole.UNRESOLVED
    return (
        CompactDataclassFieldRole.STORED_INIT
        if init_value.value
        else CompactDataclassFieldRole.STORED_NON_INIT
    )
