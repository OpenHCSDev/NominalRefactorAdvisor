# Extracted from 580b14d4:nominal_refactor_advisor/class_identity_observations.py


class IdentityPositions:
    """Which expressions of one module are identity positions, and what each says."""

    MULTIWORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z0-9]+")
    IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
    CLASS_NAME_ATTRIBUTES = frozenset(("__name__", "__qualname__"))
    COLLECTIONS = frozenset(
        (
            "tuple",
            "list",
            "set",
            "frozenset",
            "Sequence",
            "MutableSequence",
            "Iterable",
            "Iterator",
            "Collection",
            "AbstractSet",
            "MutableSet",
        )
    )

    def __init__(self, parsed_module: ParsedModule) -> None:
        self.parsed_module = parsed_module
        self.index: ModuleSyntaxIndex = module_syntax_index(parsed_module.module)
        self.node_indices = {
            id(node): node_index
            for node_index, node in enumerate(self.index.depth_first_nodes)
        }
        self.references = ModuleReferenceScope(
            parsed_module.module, package=self.package_of(parsed_module)
        )

    @staticmethod
    def package_of(parsed_module: ParsedModule) -> str:
        if parsed_module.is_package_init:
            return parsed_module.module_name
        return parsed_module.module_name.rpartition(".")[0]

    @staticmethod
    def site(
        node: ast.expr, detail: tuple[str, ...]
    ) -> tuple[ast.expr, tuple[str, ...]]:
        return node, detail

    def sites(self) -> Iterator[RestatingSite]:
        for node, detail in self.spellings():
            yield RestatingSiteFamily.site(
                self.parsed_module,
                node,
                self.index.lexical_symbol(self.node_indices[id(node)]),
                detail[-1],
                detail,
            )

    @cached_property
    def class_bindings(self) -> dict[str, tuple[int, str]]:
        """Each name a top-level class statement or `from` import binds, and nothing else does.

        The value is the binding statement's index in the module body and the
        qualified name it binds; a name bound anywhere else too (a parameter, a
        local, a second import) is left out, so no scope can shadow it.
        """

        counts = Counter(
            bound
            for node in ast.walk(self.parsed_module.module)
            for bound in ModuleReferenceScope.bound_names(node)
        )
        imports = module_import_aliases(self.parsed_module)
        found: dict[str, tuple[int, str]] = {}
        for index, statement in enumerate(self.parsed_module.module.body):
            match statement:
                case ast.ClassDef(name=name):
                    found[name] = (index, f"{self.parsed_module.module_name}.{name}")
                case ast.ImportFrom(names=aliases):
                    for alias in aliases:
                        bound = ImportBoundNameProjection(statement).alias_bound_name(alias)
                        if bound in imports:
                            found[bound] = (index, imports[bound])
        return {name: bound for name, bound in found.items() if counts[name] == 1}

    @cached_property
    def statement_indices(self) -> dict[int, int]:
        return {
            id(statement): index
            for index, statement in enumerate(self.parsed_module.module.body)
        }

    @cached_property
    def named_anywhere(self) -> frozenset[str]:
        """Every name the module binds or reads anywhere."""

        return frozenset(
            name
            for node in ast.walk(self.parsed_module.module)
            for name in (
                *ModuleReferenceScope.bound_names(node),
                *((node.id,) if isinstance(node, ast.Name) else ()),
            )
        )

    @cached_property
    def star_imported(self) -> bool:
        """Whether a `from module import *` binds names this module cannot list."""

        return any(
            alias.name == DeclaredIdentityFamily.STAR
            for node in ast.walk(self.parsed_module.module)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        )

    def class_reference(self, node_index: int, literal: str) -> tuple[str, str]:
        """What the literal's name is where the literal runs: its binding, and where an import could go.

        The first is the qualified name a top-level class statement or import
        binds it to, when that runs before the top-level statement holding the
        literal, so the name is bound whenever the literal is evaluated. The
        second, when the file binds and reads the name nowhere and it is no
        builtin, is
        that holding statement's index: a `from` statement before it could
        import the class. Both are empty otherwise, and in a module with a star
        import, whose bindings no source lists.
        """

        holding = next(
            (
                index
                for ancestor in self.index.ancestor_nodes(node_index)
                if (index := self.statement_indices.get(id(ancestor))) is not None
            ),
            None,
        )
        if holding is None or self.star_imported:
            return "", ""
        bound = self.class_bindings.get(literal)
        if bound is not None:
            return (bound[1] if bound[0] < holding else ""), ""
        if literal in self.named_anywhere or hasattr(builtins, literal):
            return "", ""
        return "", str(holding)

    @classmethod
    def multiword(cls, literal: str) -> bool:
        return (
            cls.IDENTIFIER.fullmatch(literal) is not None
            and len(cls.MULTIWORD.findall(literal)) >= 2
        )

    @staticmethod
    def string(node: ast.AST) -> str | None:
        match node:
            case ast.Constant(value=str() as value):
                return value
        return None

    called = staticmethod(ModuleReferenceScope.called)

    def spellings(self) -> Iterator[tuple[ast.expr, tuple[str, ...]]]:
        yield from self.type_sites()
        for node_index, node in self.index.indexed_nodes_of_type(ast.Compare):
            yield from self.compare_sites(node_index, node)
        for node_index, node in self.index.indexed_nodes_of_type(ast.Match):
            yield from self.match_sites(node_index, node)
        for node_index, node in self.index.indexed_nodes_of_type(ast.Call):
            yield from self.call_sites(node_index, node)
        for node_index, node in self.index.indexed_nodes_of_type(ast.Return):
            try:
                value = StatementValue.of(node).require_value("a bare return passes nothing")
            except AbsentStatementValue:
                continue
            yield from self.passed_sites(node_index, value, "return")
        yield from self.module_sites()

    # Type positions -----------------------------------------------------------

    def type_sites(self) -> Iterator[tuple[ast.expr, tuple[str, ...]]]:
        """Class names in quoted type strings: a reference wherever the name would resolve.

        Where it resolves on any interpreter the site is a `type` site; an
        annotation that resolves only when evaluated lazily is an `annotation`
        site, which the join keeps only for projects whose Python is lazy.
        """

        for quoted in self.references.quoted_type_names():
            rewrite = self.references.rewrite_of(quoted)
            for name, root in quoted.names:
                if self.references.resolves_eagerly(
                    root, quoted.node.lineno, quoted.annotation
                ):
                    yield self.site(quoted.node, TypeSpelling.detail(rewrite, name))
                elif quoted.annotation:
                    yield self.site(
                        quoted.node, LazyAnnotationSpelling.detail(rewrite, name)
                    )

    # Comparison and match positions ---------------------------------------------

    def compared(
        self, node_index: int, subject: ast.expr, literal_node: ast.expr
    ) -> Iterator[tuple[ast.expr, tuple[str, ...]]]:
        literal = self.string(literal_node)
        if literal is None or isinstance(subject, ast.Constant):
            return
        match subject:
            case ast.Attribute(attr=attribute) if (
                attribute in self.CLASS_NAME_ATTRIBUTES
            ):
                yield self.site(
                    literal_node,
                    ClassNameSpelling.detail(
                        *self.class_reference(node_index, literal), literal
                    ),
                )
                return
            case ast.Attribute(value=receiver, attr=attribute):
                yield self.site(
                    literal_node,
                    MemberSpelling.detail(
                        attribute,
                        *self.receiver(node_index, receiver).detail(),
                        literal,
                    ),
                )
        if self.multiword(literal):
            yield self.site(
                literal_node,
                ComparedWordSpelling.detail(
                    *self.class_reference(node_index, literal), literal
                ),
            )

    def compare_sites(
        self, node_index: int, node: ast.Compare
    ) -> Iterator[tuple[ast.expr, tuple[str, ...]]]:
        left = node.left
        for operator, right in zip(node.ops, node.comparators):
            match operator, right:
                case ast.Eq() | ast.NotEq() | ast.Is() | ast.IsNot(), _:
                    yield from self.compared(node_index, left, right)
                    yield from self.compared(node_index, right, left)
                case ast.In() | ast.NotIn(), ast.Set(elts=items) | ast.Tuple(
                    elts=items
                ) | ast.List(elts=items):
                    for item in items:
                        yield from self.compared(node_index, left, item)
            left = right

    def match_sites(
        self, node_index: int, node: ast.Match
    ) -> Iterator[tuple[ast.expr, tuple[str, ...]]]:
        for case in node.cases:
            for literal_node in self.pattern_literals(case.pattern):
                yield from self.compared(node_index, node.subject, literal_node)

    @classmethod
    def pattern_literals(cls, pattern: ast.pattern) -> Iterator[ast.expr]:
        match pattern:
            case ast.MatchValue(value=ast.Constant(value=str()) as literal):
                yield literal
            case ast.MatchOr(patterns=alternatives):
                for alternative in alternatives:
                    yield from cls.pattern_literals(alternative)

    # Argument and return positions -----------------------------------------------

    def call_sites(
        self, node_index: int, node: ast.Call
    ) -> Iterator[tuple[ast.expr, tuple[str, ...]]]:
        for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
            yield from self.passed_sites(node_index, argument, "argument")
        match node:
            case ast.Call(
                func=ast.Attribute(value=receiver, attr=method),
                args=[ast.Constant(value=str() as literal) as literal_node, *_],
            ) if isinstance(receiver, (ast.Name, ast.Attribute)):
                yield self.site(
                    literal_node,
                    LookupSpelling.detail(self.called(receiver), method, literal),
                )

    def passed_sites(
        self, node_index: int, value: ast.expr, position: str
    ) -> Iterator[tuple[ast.expr, tuple[str, ...]]]:
        for passed in self.passed_values(value):
            literal = self.string(passed)
            if literal is not None and self.multiword(literal):
                yield self.site(
                    passed,
                    WordSpelling.detail(
                        position, *self.class_reference(node_index, literal), literal
                    ),
                )
            match passed:
                case ast.Attribute(value=receiver, attr=attribute) if not self.own(
                    node_index, receiver
                ):
                    yield self.site(
                        passed,
                        FlattenedSpelling.detail(
                            attribute, *self.receiver(node_index, receiver).detail()
                        ),
                    )

    def receiver(self, node_index: int, receiver: ast.expr) -> Receiver:
        """The receiver's text, last segment, and the type its binding declares for it."""

        class_object = instance = ""
        if isinstance(receiver, ast.Name):
            declared = self.declared_type(node_index, receiver.id)
            match declared:
                case ast.Subscript(value=ast.Name(id="type"), slice=element):
                    class_object = self.called(element)
                case ast.Name() | ast.Attribute():
                    instance = self.called(declared)
        return Receiver(
            ast.unparse(receiver), self.called(receiver), class_object, instance
        )

    def declared_type(self, node_index: int, name: str) -> ast.expr | None:
        """The type the nearest binding of `name` around this node declares.

        A parameter declares its annotation; a `for` or comprehension target
        declares the element type of what it iterates, when that is a parameter
        annotated as a collection (or a `zip` of such), unpacked by the target.
        Nothing wider: no assignment or flow is followed.
        """

        for ancestor in reversed(self.index.ancestor_nodes(node_index)):
            for target, iterated in self.loops(ancestor):
                if name in self.target_names(target):
                    return self.unpacked(
                        target,
                        self.elements(self.node_indices[id(ancestor)], iterated),
                        name,
                    )
            match ancestor:
                case ast.FunctionDef(args=arguments) | ast.AsyncFunctionDef(
                    args=arguments
                ):
                    for argument in (
                        *arguments.posonlyargs,
                        *arguments.args,
                        *arguments.kwonlyargs,
                    ):
                        if argument.arg == name:
                            return argument.annotation
        return None

    @staticmethod
    def loops(node: ast.AST) -> Iterator[tuple[ast.expr, ast.expr]]:
        match node:
            case ast.For(target=target, iter=iterated) | ast.AsyncFor(
                target=target, iter=iterated
            ):
                yield target, iterated
            case (
                ast.ListComp(generators=generators)
                | ast.SetComp(generators=generators)
                | ast.GeneratorExp(generators=generators)
                | ast.DictComp(generators=generators)
            ):
                for generator in generators:
                    yield generator.target, generator.iter

    @staticmethod
    def target_names(target: ast.expr) -> frozenset[str]:
        return frozenset(
            node.id for node in ast.walk(target) if isinstance(node, ast.Name)
        )

    def elements(self, node_index: int, iterated: ast.expr) -> ast.expr | None:
        """The element type of an iterated parameter, or a tuple of them for `zip`."""

        match iterated:
            case ast.Name(id=name):
                return self.collection_element(self.declared_type(node_index, name))
            case ast.Call(func=ast.Name(id="zip"), args=arguments):
                return ast.Tuple(
                    elts=[
                        self.elements(node_index, argument) or ast.Constant(None)
                        for argument in arguments
                    ]
                )
        return None

    @classmethod
    def collection_element(cls, annotation: ast.expr | None) -> ast.expr | None:
        match annotation:
            case ast.Subscript(
                value=collection,
                slice=ast.Tuple(elts=[element, ast.Constant(value=builtins_ellipsis)]),
            ) if (
                builtins_ellipsis is Ellipsis
                and cls.called(collection) in cls.COLLECTIONS
            ):
                return element
            case ast.Subscript(value=collection, slice=element) if cls.called(
                collection
            ) in cls.COLLECTIONS and not isinstance(element, ast.Tuple):
                return element
        return None

    @classmethod
    def unpacked(
        cls, target: ast.expr, element: ast.expr | None, name: str
    ) -> ast.expr | None:
        match target, element:
            case ast.Name(id=bound), _ if bound == name:
                return element
            case ast.Tuple(elts=targets), ast.Tuple(elts=elements) | ast.Subscript(
                value=ast.Name(id="tuple"), slice=ast.Tuple(elts=elements)
            ) if len(targets) == len(elements):
                for inner, inner_element in zip(targets, elements):
                    if name in cls.target_names(inner):
                        return cls.unpacked(inner, inner_element, name)
        return None

    @classmethod
    def passed_values(cls, value: ast.expr) -> Iterator[ast.expr]:
        """The value itself, or the items of a literal tuple, list, set or boolean chain."""

        match value:
            case (
                ast.Tuple(elts=items)
                | ast.List(elts=items)
                | ast.Set(elts=items)
                | ast.BoolOp(values=items)
            ):
                for item in items:
                    yield from cls.passed_values(item)
            case _:
                yield value

    def own(self, node_index: int, receiver: ast.expr) -> bool:
        """Whether the receiver is the enclosing method's own object or class."""

        if not isinstance(receiver, ast.Name):
            return False
        for ancestor in reversed(self.index.ancestor_nodes(node_index)):
            match ancestor:
                case ast.ClassDef(name=name):
                    return receiver.id == name
                case ast.FunctionDef(args=arguments) | ast.AsyncFunctionDef(
                    args=arguments
                ):
                    parameters = [*arguments.posonlyargs, *arguments.args]
                    if parameters and parameters[0].arg == receiver.id:
                        return True
        return False

    # Module positions ---------------------------------------------------------

    def module_sites(self) -> Iterator[tuple[ast.expr, tuple[str, ...]]]:
        for spelled in SpelledModules(self.parsed_module.module):
            yield self.site(
                spelled.literal,
                ModuleSpelling.detail(
                    self.references.module_rewrite(spelled), spelled.name
                ),
            )
