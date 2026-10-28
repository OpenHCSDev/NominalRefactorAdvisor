# Extracted from 310b9047:nominal_refactor_advisor/quoted_reference.py


class ModuleReferenceScope:
    """What one module binds, when, and where it spells types as strings."""

    FUTURE = "__future__"
    LAZY_ANNOTATIONS = "annotations"
    TYPE_VARIABLE = "TypeVar"
    CAST = "cast"
    ANNOTATED = "Annotated"
    TYPE_METADATA = frozenset((ANNOTATED, "Literal"))
    OWN_PACKAGE = "__package__"

    def __init__(self, module: ast.Module, package: str = "") -> None:
        self.module = module
        self.package = package

    @cached_property
    def lazy_annotations(self) -> bool:
        """Whether the module imports `annotations` from `__future__`."""

        return any(
            isinstance(statement, ast.ImportFrom)
            and statement.module == self.FUTURE
            and any(alias.name == self.LAZY_ANNOTATIONS for alias in statement.names)
            for statement in self.module.body
        )

    @cached_property
    def bound_after(self) -> dict[str, int]:
        """The line after which each module-level name is bound at run time.

        A class or function is bound when its statement ends, so a reference
        inside its own body still runs before the name exists. Names bound only
        under a condition (`if TYPE_CHECKING:`) are not bound here.
        """

        bound: dict[str, int] = {}
        for statement in self.module.body:
            match statement:
                case (
                    ast.ClassDef(name=name)
                    | ast.FunctionDef(name=name)
                    | ast.AsyncFunctionDef(name=name)
                ):
                    bound.setdefault(name, statement.end_lineno or statement.lineno)
                case ast.Import() | ast.ImportFrom():
                    for name in ImportBoundNameProjection(statement).names():
                        bound.setdefault(name, statement.lineno)
                case ast.Assign(targets=targets):
                    for target in targets:
                        if isinstance(target, ast.Name):
                            bound.setdefault(target.id, statement.lineno)
                case ast.AnnAssign(target=ast.Name(id=name), value=value) if (
                    value is not None
                ):
                    bound.setdefault(name, statement.lineno)
        return bound

    def resolves_eagerly(self, root: str, line: int, annotation: bool) -> bool:
        """Whether the name resolves unquoted at this position on any interpreter.

        A future-import annotation is never evaluated eagerly; any other position
        can name what the module has bound before it runs, or a builtin.
        """

        if annotation and self.lazy_annotations:
            return True
        bound = self.bound_after.get(root)
        if bound is not None:
            return bound < line
        return hasattr(builtins, root)

    def resolves(
        self, root: str, line: int, annotation: bool, interpreter: DeclaredInterpreter
    ) -> bool:
        """Whether the name resolves unquoted at this position on the declared interpreter."""

        if self.resolves_eagerly(root, line, annotation):
            return True
        return annotation and ReferenceRewrite.LAZY_INTERPRETER.rewritable(interpreter)

    def quoted_type_names(self) -> Iterator[QuotedTypeName]:
        for expression, annotation in self.type_expressions():
            metadata = {
                id(part)
                for node in ast.walk(expression)
                if isinstance(node, ast.Subscript)
                and self.called(node.value) in self.TYPE_METADATA
                for part in self.metadata_parts(node)
            }
            for node in ast.walk(expression):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in metadata
                ):
                    names = self.annotation_names(node.value)
                    if names:
                        yield QuotedTypeName(node, names, annotation)

    def rewrite_of(self, quoted: QuotedTypeName) -> ReferenceRewrite:
        """Where the whole quoted string can be written as the expression it spells."""

        if any(
            isinstance(part, ast.Constant) and isinstance(part.value, str)
            for part in ast.walk(quoted.expression)
        ):
            return ReferenceRewrite.NONE
        if all(
            self.resolves_eagerly(root, quoted.node.lineno, quoted.annotation)
            for _name, root in quoted.names
        ):
            return ReferenceRewrite.ANY_INTERPRETER
        if quoted.annotation:
            return ReferenceRewrite.LAZY_INTERPRETER
        return ReferenceRewrite.NONE

    def unquotable(self, interpreter: DeclaredInterpreter) -> Iterator[QuotedTypeName]:
        """The quoted type strings that can be written as the expressions they spell."""

        for quoted in self.quoted_type_names():
            if self.rewrite_of(quoted).rewritable(interpreter):
                yield quoted

    def module_rewrite(self, spelled: SpelledModule) -> ReferenceRewrite:
        """Whether the spelled module can be written as a reference the file already has."""

        if self.module_reference(spelled) is not None:
            return ReferenceRewrite.ANY_INTERPRETER
        return ReferenceRewrite.NONE

    def module_reference(self, spelled: SpelledModule) -> str | None:
        """The expression that stands where the spelled module is, or None.

        A module the file imports eagerly is its import's reference. The file's
        own package, where the string itself is the value, is `__package__`:
        a module with a top-level relative import runs only as a submodule of
        its package, and the import system sets `__package__` to that package's
        name, so `__package__` is the string wherever the file does not
        rebind it.
        """

        if spelled.replaced is None:
            return None
        if spelled.name in self.imported_modules:
            return f"{self.imported_modules[spelled.name]}{spelled.suffix}"
        if (
            spelled.replaced is spelled.literal
            and self.package
            and spelled.name == self.package
            and self.runs_in_package
        ):
            return self.OWN_PACKAGE
        return None

    @cached_property
    def runs_in_package(self) -> bool:
        """Whether the module imports relatively at top level and never rebinds `__package__`."""

        return any(
            isinstance(statement, ast.ImportFrom) and statement.level > 0
            for statement in self.module.body
        ) and not any(
            bound == self.OWN_PACKAGE
            for node in ast.walk(self.module)
            for bound in self.bound_names(node)
        )

    def type_expressions(self) -> Iterator[tuple[ast.expr, bool]]:
        """Each type position, and whether it is an annotation."""

        for node in ast.walk(self.module):
            match node:
                case ast.FunctionDef(
                    args=arguments, returns=returns
                ) | ast.AsyncFunctionDef(args=arguments, returns=returns):
                    if returns is not None:
                        yield returns, True
                    for argument in (
                        *arguments.posonlyargs,
                        *arguments.args,
                        *arguments.kwonlyargs,
                        *filter(None, (arguments.vararg, arguments.kwarg)),
                    ):
                        if argument.annotation is not None:
                            yield argument.annotation, True
                case ast.AnnAssign(annotation=annotation):
                    yield annotation, True
                case ast.ClassDef(bases=bases):
                    for base in bases:
                        yield base, False
                case ast.Call(func=callee, args=[first, *rest], keywords=keywords):
                    called = self.called(callee)
                    if called == self.TYPE_VARIABLE:
                        for bound in (*rest, *(keyword.value for keyword in keywords)):
                            yield bound, False
                    elif called == self.CAST:
                        yield first, False

    @staticmethod
    def called(node: ast.expr) -> str:
        match node:
            case ast.Name(id=name) | ast.Attribute(attr=name):
                return name
        return ""

    def metadata_parts(self, node: ast.Subscript) -> Iterator[ast.AST]:
        """`Literal[...]` is all values; `Annotated[T, ...]` is T and then metadata."""

        elements = (
            node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        )
        kept = 1 if self.called(node.value) == self.ANNOTATED else 0
        for element in elements[kept:]:
            yield from ast.walk(element)

    @staticmethod
    def annotation_names(literal: str) -> tuple[tuple[str, str], ...]:
        """The names a quoted annotation refers to, each with the module-level name it starts from."""

        try:
            parsed = ast.parse(literal, mode="eval")
        except SyntaxError:
            return ()
        found: dict[tuple[str, str], None] = {}
        inner = {
            id(node.value)
            for node in ast.walk(parsed)
            if isinstance(node, ast.Attribute)
        }
        for node in ast.walk(parsed):
            match node:
                case ast.Name(id=name) if id(node) not in inner:
                    found[(name, name)] = None
                case ast.Attribute(attr=name) if id(node) not in inner:
                    found[(name, ModuleReferenceScope.root_name(node))] = None
        return tuple(name for name in found if name[1])

    @staticmethod
    def root_name(node: ast.expr) -> str:
        """The name an attribute chain starts from, or empty if it starts elsewhere."""

        match node:
            case ast.Attribute(value=value):
                return ModuleReferenceScope.root_name(value)
            case ast.Name(id=name):
                return name
        return ""

    @cached_property
    def imported_modules(self) -> dict[str, str]:
        """Each module the module imports eagerly by `import`, and the expression that names it.

        `import a.b` makes `a` and `a.b` reachable as `a` and `a.b`; `import a.b as
        c` makes `a.b` reachable as `c`. A root name the module also binds any
        other way names something else somewhere, so it reaches nothing here.
        """

        reachable: dict[str, str] = {}
        bindings: dict[str, int] = {}
        for node in ast.walk(self.module):
            for bound in self.bound_names(node):
                bindings[bound] = bindings.get(bound, 0) + 1
        for statement in self.module.body:
            if not isinstance(statement, ast.Import):
                continue
            for alias in statement.names:
                if ImportAlias.from_alias(alias).spells_as_clause:
                    reachable[alias.name] = ImportBoundNameProjection(statement).alias_bound_name(alias)
                    continue
                parts = alias.name.split(".")
                for stop in range(1, len(parts) + 1):
                    dotted = ".".join(parts[:stop])
                    reachable[dotted] = dotted
        aliased = set(reachable.values()) - set(reachable)
        return {
            dotted: expression
            for dotted, expression in reachable.items()
            if bindings.get(expression.partition(".")[0], 0)
            == (1 if expression in aliased else 0)
        }

    @staticmethod
    def bound_names(node: ast.AST) -> Iterator[str]:
        match node:
            case ast.Import(names=aliases):
                for alias in aliases:
                    if ImportAlias.from_alias(alias).spells_as_clause:
                        yield ImportBoundNameProjection(node).alias_bound_name(alias)
            case ast.ImportFrom():
                yield from ImportBoundNameProjection(node).names()
            case ast.Name(id=name, ctx=ast.Store() | ast.Del()):
                yield name
            case (
                ast.FunctionDef(name=name)
                | ast.AsyncFunctionDef(name=name)
                | ast.ClassDef(name=name)
            ):
                yield name
            case ast.arg(arg=name):
                yield name
