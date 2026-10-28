# Extracted from ac2a002d:nominal_refactor_advisor/manual_registry.py


@dataclass(frozen=True)
class AutoRegisterInstanceViewComponent(RegistryEntries):
    """One constructor-valued view proved against its AutoRegister authority."""

    module: ast.Module
    authority_node: ast.ClassDef
    assignment: RegistryAssignment
    entries: tuple[SourceClassKeyEntry, ...]

    def require_original_entry_values(
        self, environment: NativeReferenceEnvironment
    ) -> None:
        self.authority.require_native_metaclass(environment)
        super().require_original_entry_values(environment)

    @classmethod
    def from_module_authority(
        cls,
        module: ast.Module,
        authority_name: str,
    ) -> AutoRegisterInstanceViewComponent:
        class_graph = DirectModuleClassGraph(module)
        authority_node = class_graph.classes_by_name.get(authority_name)
        if authority_node is None:
            raise ValueError(
                f"AutoRegister authority {authority_name!r} is not a top-level class"
            )
        candidates = tuple(
            candidate
            for statement in module.body
            if (
                candidate := cls.from_assignment(
                    module,
                    authority_node,
                    statement,
                    class_graph,
                )
            )
            is not None
        )
        if len(candidates) != 1:
            raise ValueError(
                f"AutoRegister authority {authority_name!r} must identify exactly "
                "one constructor-valued instance view"
            )
        component = candidates[0]
        component.require_complete(class_graph)
        return component

    @classmethod
    def from_assignment(
        cls,
        module: ast.Module,
        authority_node: ast.ClassDef,
        statement: ast.stmt,
        class_graph: DirectModuleClassGraph,
    ) -> AutoRegisterInstanceViewComponent | None:
        if not isinstance(statement, RegistryAssignment):
            return None
        try:
            assignment_name = ExpressionReference.decoded(
                _assignment_target(statement)
            ).identifier
        except ValueError:
            return None
        value = _assignment_value(statement)
        if (
            assignment_name is None
            or not isinstance(value, ast.Dict)
            or len(value.keys) < 2
            or any(key is None for key in value.keys)
        ):
            return None
        entries = []
        for key, value_node in zip(value.keys, value.values, strict=True):
            if (
                key is None
                or not isinstance(value_node, ast.Call)
                or value_node.args
                or value_node.keywords
                or not isinstance(value_node.func, ast.Name)
            ):
                return None
            class_node = class_graph.classes_by_name.get(value_node.func.id)
            if class_node is None or not class_graph.descends_from(
                class_node,
                authority_node.name,
            ):
                return None
            entries.append(ConstructorClassKeyEntry(class_node, key, value_node))
        return cls(
            module=module,
            authority_node=authority_node,
            assignment=statement,
            entries=tuple(entries),
        )

    @cached_property
    def authority(self) -> AutoRegisterClassAuthority:
        return AutoRegisterClassAuthority(self.authority_node)

    @property
    def authority_name(self) -> str:
        return self.authority_node.name

    @property
    def assignment_name(self) -> str:
        assignment_name = ExpressionReference.decoded(
            _assignment_target(self.assignment)
        ).identifier
        if assignment_name is None:
            raise ValueError("Instance-view assignment target is not a name")
        return assignment_name

    @property
    def registry_key_attribute(self) -> str:
        registry_key_attribute = self.authority.registry_key_attribute
        if registry_key_attribute is None:
            raise ValueError(
                f"AutoRegister authority {self.authority_name!r} has no registry key"
            )
        return registry_key_attribute

    def require_complete(self, class_graph: DirectModuleClassGraph) -> None:
        if not self.authority.keyed_metaclass_family:
            raise ValueError(
                f"{self.authority_name!r} declares no metaclass and registry key"
            )
        if self.authority.declares_registry:
            registry_value = self.authority.assignment_value(REGISTRY_ATTRIBUTE_NAME)
            if not (
                isinstance(registry_value, ast.Dict)
                and not registry_value.keys
                and not registry_value.values
            ):
                raise ValueError(
                    f"AutoRegister authority {self.authority_name!r} must own an "
                    "empty direct registry"
                )
        self.require_ordered_unique_entries()
        assignment_line = self.assignment.lineno
        if any(
            (node.end_lineno or node.lineno) >= assignment_line
            for node in self.class_nodes
        ):
            raise ValueError(
                "Instance-view assignment must follow every constructed class"
            )
        registered_names = frozenset(self.class_names)
        unsafe_descendants = tuple(
            node.name
            for node in class_graph.descendants_of(self.authority_name)
            if node.lineno < assignment_line
            and node.name not in registered_names
            and (
                any(
                    class_graph.descends_from(node, registered_name)
                    for registered_name in registered_names
                )
                or _class_declares_non_null_name(
                    node,
                    self.registry_key_attribute,
                )
            )
        )
        if unsafe_descendants:
            raise ValueError(
                f"Instance view omits registered descendants {unsafe_descendants!r}"
            )
