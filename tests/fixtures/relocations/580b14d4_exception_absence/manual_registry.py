# Extracted from 580b14d4:nominal_refactor_advisor/manual_registry.py


@dataclass(frozen=True)
class SourceModuleMappingKeyAuthority(MappingKeyResolverABC[ast.Module]):
    """Resolve AST key expressions from their exact module-local declarations."""

    module: ast.Module

    @cached_property
    def parsed_module(self) -> ParsedModule:
        return ParsedModule(
            Path("__nra_registry_key__.py"),
            "__nra_registry_key__",
            False,
            self.module,
            ast.unparse(self.module),
        )

    @cached_property
    def enum_declarations_by_name(
        self,
    ) -> dict[str, PythonEnumDeclarationAuthority]:
        bindings = ModuleNominalBindingAuthority(self.parsed_module)
        declarations: dict[str, PythonEnumDeclarationAuthority] = {}
        for name, node in DirectModuleClassGraph(self.module).classes_by_name.items():
            snapshot = bindings.snapshot_before(node.lineno)

            def qualified_reference(
                reference: ast.expr,
                shadowed_names: frozenset[str],
            ) -> str | None:
                parts = ExpressionReference.decoded(reference).attribute_chain
                if not parts or parts[0] in shadowed_names:
                    return None
                try:
                    return snapshot.reference_for(parts).require_bound_name()
                except UnboundReference:
                    # QualifiedReferenceResolver's str | None protocol.
                    return None

            try:
                declarations[name] = PYTHON_ENUM_BASE_AUTHORITY.declaration_from_node(
                    node,
                    identity=f"{self.parsed_module.module_name}.{name}",
                    qualified_reference=qualified_reference,
                )
            except ValueError:
                continue
        return declarations

    def _lexical_mapping_key(
        self,
        reference: LexicalValueReference,
        context: ast.Module,
    ) -> Hashable:
        if context is not self.module or len(reference.attribute_path) != 1:
            raise ValueError("Mapping key reference has no exact module declaration")
        declaration = self.enum_declarations_by_name.get(reference.root_name)
        if declaration is None:
            raise ValueError("Mapping key declaration semantics remain unproved")
        return declaration.require_mapping_key(reference.attribute_path[0])


def _class_declares_non_null_name(node: ast.ClassDef, name: str) -> bool:
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            if not any(
                ExpressionReference.decoded(target).identifier == name
                for target in statement.targets
            ):
                continue
            value = statement.value
        elif (
            isinstance(statement, ast.AnnAssign)
            and ExpressionReference.decoded(statement.target).identifier == name
        ):
            try:
                value = StatementValue.of(statement).require_value("a bare annotation")
            except AbsentStatementValue:
                return True
        else:
            continue
        return not CompactValueExpression.project(value).value_is_none_literal
    return False
