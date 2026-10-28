# Extracted from 580b14d4:nominal_refactor_advisor/observation_families.py


@dataclass(frozen=True)
class TypeArgumentResolver:
    """Resolve an `isinstance` type argument to qualified class names, fail closed.

    Names resolve through the module's final import and builtin bindings; a
    tuple, a `|` union, or a once-assigned module-level alias of either expands
    to its members. Anything unresolved or shadowed in an enclosing function
    makes the whole site unresolved.
    """

    parsed_module: ParsedModule

    @cached_property
    def snapshot(self) -> ModuleNominalBindingSnapshot:
        return ModuleNominalBindingAuthority(self.parsed_module).snapshot_before(None)

    @cached_property
    def aliases(self) -> dict[str, ast.expr]:
        assignments = [
            (statement.targets[0].id, statement.value)
            for statement in self.parsed_module.module.body
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ]
        counts = Counter(name for name, _value in assignments)
        return {
            name: value
            for name, value in assignments
            if counts[name] == 1 and self._is_member_collection(value)
        }

    @staticmethod
    def _is_member_collection(node: ast.AST) -> bool:
        match node:
            case ast.Tuple() | ast.BinOp(op=ast.BitOr()):
                return True
        return False

    def members(
        self,
        node: ast.expr,
        shadowed: frozenset[str],
        expanding: frozenset[str] = frozenset(),
    ) -> tuple[ast.expr, ...]:
        match node:
            case ast.Tuple(elts=elements):
                return tuple(
                    member
                    for element in elements
                    for member in self.members(element, shadowed, expanding)
                )
            case ast.BinOp(left=left, op=ast.BitOr(), right=right):
                return (
                    *self.members(left, shadowed, expanding),
                    *self.members(right, shadowed, expanding),
                )
            case ast.Name(id=name) if (
                name in self.aliases and name not in shadowed | expanding
            ):
                return self.members(self.aliases[name], shadowed, expanding | {name})
        return (node,)

    def qualified_name(self, member: ast.expr, shadowed: frozenset[str]) -> str | None:
        if ExpressionReference.decoded(member).qualified_name is None:
            return None
        parts = ExpressionReference.decoded(member).attribute_chain
        if parts is None or parts[0] in shadowed:
            return None
        try:
            return self.snapshot.reference_for(parts).require_bound_name()
        except UnboundReference:
            # The str | None resolver protocol: unbound joins its ast-shaped absences.
            return None

    def qualified_class_at(
        self, member: ast.expr, ancestors: Sequence[ast.AST]
    ) -> str | None:
        """Qualified class a name expression denotes beneath these ancestors."""

        return self.qualified_name(member, _enclosing_function_names(ancestors))

    def qualified_names(
        self, type_argument: ast.expr, shadowed: frozenset[str]
    ) -> tuple[str, ...] | None:
        members = self.members(type_argument, shadowed)
        names = tuple(self.qualified_name(member, shadowed) for member in members)
        if not names or None in names:
            return None
        return tuple(sorted(set(names)))

    def qualified_names_of_test(self, call: ast.Call) -> tuple[str, ...] | None:
        """Qualified classes one `isinstance` call of this module tests, if resolved."""

        arguments = BuiltinCallName.isinstance_arguments(call)
        if arguments is None:
            return None
        parents = module_syntax_index(self.parsed_module.module).parent_by_node
        ancestors = tuple(self._ancestors(parents, call))
        return self.qualified_names(arguments[1], _enclosing_function_names(ancestors))

    @staticmethod
    def _ancestors(
        parents: Mapping[ast.AST, ast.AST], node: ast.AST
    ) -> Iterator[ast.AST]:
        parent = parents.get(node)
        if parent is not None:
            yield parent
            yield from TypeArgumentResolver._ancestors(parents, parent)
