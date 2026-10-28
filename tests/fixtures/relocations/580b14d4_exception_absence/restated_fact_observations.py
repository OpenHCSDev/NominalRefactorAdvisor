# Extracted from 580b14d4:nominal_refactor_advisor/restated_fact_observations.py


class LegacyMarkerSiteFamily(RestatingSiteFamily):
    """A path that says it is the old one, beside what replaces it.

    Three self-declarations count: a deprecation mechanism (a `warn` of a
    deprecation class, a deprecation decorator), prose in the docstring's summary line, and an
    identifier with a marker word *beside a sibling of the same stem* (the old
    path kept next to its replacement), or an alias `old = new` of a replacement.
    A name that merely contains a marker word, with no sibling, is not one.
    """

    @classmethod
    def sites(cls, parsed_module: ParsedModule) -> Iterator[RestatingSite]:
        index = module_syntax_index(parsed_module.module)
        for node_type in (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef):
            for node_index, node in index.indexed_nodes_of_type(node_type):
                yield from cls.declaration(parsed_module, index, node_index, node)
        for node_index, node in index.indexed_nodes_of_type(ast.arguments):
            siblings = {
                argument.arg
                for argument in (
                    *node.posonlyargs,
                    *node.args,
                    *node.kwonlyargs,
                    *([node.vararg] if node.vararg else []),
                    *([node.kwarg] if node.kwarg else []),
                )
            }
            for argument in sorted(siblings):
                if cls.has_replacement(argument, siblings):
                    argument_node = next(
                        candidate
                        for candidate in (
                            *node.posonlyargs,
                            *node.args,
                            *node.kwonlyargs,
                        )
                        if candidate.arg == argument
                    )
                    yield cls.site(
                        parsed_module,
                        argument_node,
                        cls.declared_symbol(index.lexical_symbol(node_index), argument),
                        argument,
                        ("parameter", "beside its replacement"),
                    )
        for node_index, node in (
            *index.indexed_nodes_of_type(ast.Assign),
            *index.indexed_nodes_of_type(ast.AnnAssign),
        ):
            owner = index.parent_node(node_index)
            if not isinstance(owner, (ast.Module, ast.ClassDef)):
                continue
            try:
                value = StatementValue.of(node).require_value("a bare annotation")
            except AbsentStatementValue:
                continue
            replacement = cls.aliased(value)
            for target in cls.assigned_names(node):
                if replacement is not None and LegacyWord.in_identifier(target):
                    yield cls.site(
                        parsed_module,
                        node,
                        cls.declared_symbol(
                            index.lexical_symbol(node_index), target
                        ),
                        target,
                        ("alias", f"of {replacement}"),
                    )
        for node_index, node in index.indexed_nodes_of_type(ast.alias):
            # An alias renames the old path; a plain import of a marked name is its consumer.
            if LegacyWord.in_identifier(node.name):
                continue
            for bound in ImportAlias.from_alias(node).legacy_marker_spellings():
                if LegacyWord.in_identifier(bound):
                    yield cls.site(
                        parsed_module,
                        node,
                        cls.declared_symbol(index.lexical_symbol(node_index), bound),
                        bound,
                        ("import alias", f"of {node.name}"),
                    )
        for node_index, call in index.indexed_nodes_of_type(ast.Call):
            warned = cls.warned_legacy_class(call)
            if warned is not None:
                symbol = index.lexical_symbol(node_index)
                yield cls.site(
                    parsed_module,
                    call,
                    symbol,
                    symbol.rpartition(".")[2],
                    ("warning", warned),
                )

    @classmethod
    def declaration(
        cls,
        parsed_module: ParsedModule,
        index,
        node_index: int,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    ) -> Iterator[RestatingSite]:
        parent = index.parent_node(node_index)
        siblings = {
            name
            for statement in index.children_by_node.get(parent, ())
            if isinstance(statement, ast.stmt)
            for name in cls.declared_names(statement)
        }
        why = cls.reason(node, siblings)
        if why is None:
            return
        symbol = cls.declared_symbol(index.lexical_symbol(node_index), node.name)
        yield cls.site(
            parsed_module, node, symbol, node.name, (type(node).__name__, why)
        )

    @staticmethod
    def declared_symbol(owner: str, name: str) -> str:
        """The dotted symbol of a name declared inside `owner`."""

        return name if owner == "<module>" else f"{owner}.{name}"

    @staticmethod
    def has_replacement(name: str, siblings: set[str]) -> bool:
        """Whether `name` has a marker word and a sibling of the same stem without one."""

        stem = LegacyWord.stem(name)
        return (
            LegacyWord.in_identifier(name)
            and bool(stem)
            and any(
                sibling != name
                and not LegacyWord.in_identifier(sibling)
                and LegacyWord.stem(sibling) == stem
                for sibling in siblings
            )
        )

    @classmethod
    def reason(
        cls,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        siblings: set[str],
    ) -> str | None:
        if cls.has_replacement(node.name, siblings):
            return "named so, beside its replacement"
        if any(
            LegacyWord.in_identifier(ast.unparse(decorator))
            for decorator in node.decorator_list
        ):
            return "decorated so"
        docstring = ast.get_docstring(node)
        # The summary line says what the thing is; later lines may merely discuss the idea.
        if docstring is not None and LegacyWord.in_prose(docstring.splitlines()[0]):
            return "documented so"
        return None

    @staticmethod
    def declared_names(statement: ast.stmt) -> Iterator[str]:
        match statement:
            case (
                ast.FunctionDef(name=name)
                | ast.AsyncFunctionDef(name=name)
                | ast.ClassDef(name=name)
            ):
                yield name
            case ast.Assign(targets=targets):
                for target in targets:
                    if isinstance(target, ast.Name):
                        yield target.id
            case ast.AnnAssign(target=ast.Name(id=name)):
                yield name

    @staticmethod
    def aliased(value: ast.expr) -> str | None:
        """The replacement an assignment aliases, when its value is a plain reference."""

        return (
            ast.unparse(value) if isinstance(value, (ast.Name, ast.Attribute)) else None
        )

    @staticmethod
    def assigned_names(node: ast.Assign | ast.AnnAssign) -> Iterator[str]:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                yield target.id

    @staticmethod
    def warned_legacy_class(call: ast.Call) -> str | None:
        """The warning class, named for an old path, that a `warn(...)` call raises."""

        match call.func:
            case ast.Attribute(attr="warn") | ast.Name(id="warn"):
                pass
            case _:
                return None
        for argument in (*call.args, *(keyword.value for keyword in call.keywords)):
            spelled = ast.unparse(argument)
            if isinstance(
                argument, (ast.Name, ast.Attribute)
            ) and LegacyWord.in_identifier(spelled):
                return spelled
        return None
