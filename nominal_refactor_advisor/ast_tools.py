from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParsedModule:
    path: Path
    module: ast.Module
    source: str


@dataclass(frozen=True)
class MethodShape:
    file_path: str
    class_name: str | None
    method_name: str
    lineno: int
    statement_count: int
    is_private: bool
    param_count: int
    decorators: tuple[str, ...]
    fingerprint: str
    statement_texts: tuple[str, ...]

    @property
    def symbol(self) -> str:
        if self.class_name:
            return f"{self.class_name}.{self.method_name}"
        return self.method_name


@dataclass(frozen=True)
class BuilderCallShape:
    file_path: str
    class_name: str | None
    function_name: str | None
    lineno: int
    callee_name: str
    keyword_names: tuple[str, ...]
    value_fingerprint: tuple[str, ...]
    source_arity: int
    source_name: str | None
    identity_field_names: tuple[str, ...]

    @property
    def symbol(self) -> str:
        owner = self.function_name or "<module>"
        if self.class_name:
            owner = f"{self.class_name}.{owner}"
        return f"{owner}:{self.callee_name}"


@dataclass(frozen=True)
class RegistrationShape:
    file_path: str
    lineno: int
    registry_name: str
    registered_class: str
    key_fingerprint: str
    key_expression: str
    registration_style: str

    @classmethod
    def from_assignment(
        cls,
        parsed_module: ParsedModule,
        node: ast.Assign,
        registry_name: str,
        key_fingerprint: str,
    ) -> "RegistrationShape":
        if not isinstance(node.value, ast.Name):
            raise TypeError("Registration assignment value must be a class name")
        return cls(
            file_path=str(parsed_module.path),
            lineno=node.lineno,
            registry_name=registry_name,
            registered_class=node.value.id,
            key_fingerprint=key_fingerprint,
            key_expression=ast.unparse(node.targets[0].slice)
            if isinstance(node.targets[0], ast.Subscript)
            else "...",
            registration_style="subscript_assignment",
        )

    @classmethod
    def from_registration_call(
        cls,
        parsed_module: ParsedModule,
        node: ast.Call,
        registry_name: str,
        registered_class: str,
        key_fingerprint: str,
    ) -> "RegistrationShape":
        return cls(
            file_path=str(parsed_module.path),
            lineno=node.lineno,
            registry_name=registry_name,
            registered_class=registered_class,
            key_fingerprint=key_fingerprint,
            key_expression=ast.unparse(
                node.args[1] if len(node.args) >= 2 else node.args[0]
            ),
            registration_style="registration_call",
        )

    @classmethod
    def from_decorator(
        cls,
        parsed_module: ParsedModule,
        node: ast.ClassDef,
        registry_name: str,
        key_fingerprint: str,
    ) -> "RegistrationShape":
        return cls(
            file_path=str(parsed_module.path),
            lineno=node.lineno,
            registry_name=registry_name,
            registered_class=node.name,
            key_fingerprint=key_fingerprint,
            key_expression=node.name,
            registration_style="decorator_registration",
        )

    @property
    def symbol(self) -> str:
        return f"{self.registry_name}[...] = {self.registered_class}"


@dataclass(frozen=True)
class ExportDictShape:
    file_path: str
    class_name: str | None
    function_name: str | None
    lineno: int
    key_names: tuple[str, ...]
    value_fingerprint: tuple[str, ...]
    source_arity: int
    source_name: str | None
    identity_field_names: tuple[str, ...]

    @property
    def symbol(self) -> str:
        owner = self.function_name or "<module>"
        if self.class_name:
            owner = f"{self.class_name}.{owner}"
        return f"{owner}:export-dict"


def parse_python_modules(root: Path) -> list[ParsedModule]:
    modules: list[ParsedModule] = []
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        modules.append(ParsedModule(path=path, module=ast.parse(source), source=source))
    return modules


class _ShapeNormalizer(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id="VAR", ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node = ast.arg(arg="ARG", annotation=None, type_comment=None)
        return ast.copy_location(node, node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value="STR"), node)
        if isinstance(node.value, (int, float, complex)):
            return ast.copy_location(ast.Constant(value=0), node)
        if node.value is None:
            return ast.copy_location(ast.Constant(value=None), node)
        if isinstance(node.value, bool):
            return ast.copy_location(ast.Constant(value=True), node)
        return ast.copy_location(ast.Constant(value="CONST"), node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        value = self.visit(node.value)
        new_node = ast.Attribute(value=value, attr="ATTR", ctx=node.ctx)
        return ast.copy_location(new_node, node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node = ast.FunctionDef(
            name="FUNC",
            args=self.visit(node.args),
            body=[self.visit(stmt) for stmt in node.body],
            decorator_list=[self.visit(dec) for dec in node.decorator_list],
            returns=None,
            type_comment=None,
        )
        return ast.copy_location(node, node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node = ast.AsyncFunctionDef(
            name="FUNC",
            args=self.visit(node.args),
            body=[self.visit(stmt) for stmt in node.body],
            decorator_list=[self.visit(dec) for dec in node.decorator_list],
            returns=None,
            type_comment=None,
        )
        return ast.copy_location(node, node)


def fingerprint_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    normalized = _ShapeNormalizer().visit(copy.deepcopy(node))
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, include_attributes=False)


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.__class__.__name__


def collect_method_shapes(parsed_module: ParsedModule) -> list[MethodShape]:
    methods: list[MethodShape] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []

        def _record_method(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            methods.append(
                MethodShape(
                    file_path=str(parsed_module.path),
                    class_name=self.class_stack[-1] if self.class_stack else None,
                    method_name=node.name,
                    lineno=node.lineno,
                    statement_count=len(node.body),
                    is_private=node.name.startswith("_")
                    and not node.name.startswith("__"),
                    param_count=len(node.args.args),
                    decorators=tuple(
                        _decorator_name(dec) for dec in node.decorator_list
                    ),
                    fingerprint=fingerprint_function(node),
                    statement_texts=tuple(ast.unparse(stmt) for stmt in node.body),
                )
            )

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._record_method(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._record_method(node)
            self.generic_visit(node)

    Visitor().visit(parsed_module.module)
    return methods


def collect_builder_call_shapes(parsed_module: ParsedModule) -> list[BuilderCallShape]:
    shapes: list[BuilderCallShape] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []
            self.function_stack: list[str] = []

        def _push_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._push_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._push_function(node)

        def visit_Call(self, node: ast.Call) -> None:
            shape = _builder_call_shape(
                parsed_module,
                node,
                self.class_stack[-1] if self.class_stack else None,
                self.function_stack[-1] if self.function_stack else None,
            )
            if shape is not None:
                shapes.append(shape)
            self.generic_visit(node)

    Visitor().visit(parsed_module.module)
    return shapes


def collect_registration_shapes(parsed_module: ParsedModule) -> list[RegistrationShape]:
    shapes: list[RegistrationShape] = []
    known_classes = {
        node.name
        for node in ast.walk(parsed_module.module)
        if isinstance(node, ast.ClassDef)
    }
    for node in ast.walk(parsed_module.module):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        if node.value.id not in known_classes:
            continue
        for target in node.targets:
            registry_name = _registry_target_name(target)
            if registry_name is None:
                continue
            key_fingerprint = _registration_key_fingerprint(target)
            if key_fingerprint is None:
                continue
            shapes.append(
                RegistrationShape.from_assignment(
                    parsed_module,
                    node,
                    registry_name,
                    key_fingerprint,
                )
            )
    shapes.extend(_collect_registration_call_shapes(parsed_module, known_classes))
    shapes.extend(_collect_decorator_registration_shapes(parsed_module))
    return shapes


def collect_export_dict_shapes(parsed_module: ParsedModule) -> list[ExportDictShape]:
    shapes: list[ExportDictShape] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []
            self.function_stack: list[str] = []

        def _push_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._push_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._push_function(node)

        def visit_Dict(self, node: ast.Dict) -> None:
            shape = _export_dict_shape(
                parsed_module,
                node,
                self.class_stack[-1] if self.class_stack else None,
                self.function_stack[-1] if self.function_stack else None,
            )
            if shape is not None:
                shapes.append(shape)
            self.generic_visit(node)

    Visitor().visit(parsed_module.module)
    return shapes


def _builder_call_shape(
    parsed_module: ParsedModule,
    node: ast.Call,
    class_name: str | None,
    function_name: str | None,
) -> BuilderCallShape | None:
    if function_name is None:
        return None
    keyword_pairs = [(kw.arg, kw.value) for kw in node.keywords if kw.arg is not None]
    if len(keyword_pairs) < 3:
        return None
    callee_name = _call_name(node.func)
    if callee_name is None:
        return None
    keyword_names = tuple(name for name, _ in keyword_pairs)
    value_fingerprint = tuple(
        _fingerprint_builder_value(value) for _, value in keyword_pairs
    )
    source_roots = set()
    for _, value in keyword_pairs:
        source_roots.update(_root_names(value))
    source_name = next(iter(source_roots)) if len(source_roots) == 1 else None
    identity_field_names = tuple(
        name for name, value in keyword_pairs if _leaf_name(value) == name
    )
    return BuilderCallShape(
        file_path=str(parsed_module.path),
        class_name=class_name,
        function_name=function_name,
        lineno=node.lineno,
        callee_name=callee_name,
        keyword_names=keyword_names,
        value_fingerprint=value_fingerprint,
        source_arity=len(source_roots),
        source_name=source_name,
        identity_field_names=identity_field_names,
    )


def _export_dict_shape(
    parsed_module: ParsedModule,
    node: ast.Dict,
    class_name: str | None,
    function_name: str | None,
) -> ExportDictShape | None:
    if function_name is None:
        return None
    key_pairs = [
        (key.value, value)
        for key, value in zip(node.keys, node.values, strict=False)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    ]
    if len(key_pairs) < 3 or len(key_pairs) != len(node.keys):
        return None
    key_names = tuple(name for name, _ in key_pairs)
    value_fingerprint = tuple(
        _fingerprint_builder_value(value) for _, value in key_pairs
    )
    source_roots = set()
    for _, value in key_pairs:
        source_roots.update(_root_names(value))
    if not source_roots:
        return None
    source_name = next(iter(source_roots)) if len(source_roots) == 1 else None
    identity_field_names = tuple(
        name for name, value in key_pairs if _leaf_name(value) == name
    )
    return ExportDictShape(
        file_path=str(parsed_module.path),
        class_name=class_name,
        function_name=function_name,
        lineno=node.lineno,
        key_names=key_names,
        value_fingerprint=value_fingerprint,
        source_arity=len(source_roots),
        source_name=source_name,
        identity_field_names=identity_field_names,
    )


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _leaf_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


class _BuilderValueNormalizer(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id="ROOT", ctx=node.ctx), node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value="STR"), node)
        if isinstance(node.value, (int, float, complex)):
            return ast.copy_location(ast.Constant(value=0), node)
        if isinstance(node.value, bool):
            return ast.copy_location(ast.Constant(value=True), node)
        if node.value is None:
            return ast.copy_location(ast.Constant(value=None), node)
        return ast.copy_location(ast.Constant(value="CONST"), node)


def _fingerprint_builder_value(node: ast.AST) -> str:
    normalized = _BuilderValueNormalizer().visit(copy.deepcopy(node))
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, include_attributes=False)


def _root_names(node: ast.AST) -> set[str]:
    roots: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Attribute(self, node: ast.Attribute) -> None:
            current: ast.AST = node
            while isinstance(current, ast.Attribute):
                current = current.value
            if isinstance(current, ast.Name):
                roots.add(current.id)
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            roots.add(node.id)

    Visitor().visit(node)
    return roots


def _registry_target_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    target = node.value
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _registration_key_fingerprint(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    return _fingerprint_builder_value(node.slice)


def _collect_registration_call_shapes(
    parsed_module: ParsedModule, known_classes: set[str]
) -> list[RegistrationShape]:
    shapes: list[RegistrationShape] = []
    for node in ast.walk(parsed_module.module):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"register", "add", "register_class", "register_type"}:
            continue
        registry_name = _call_name(node.func.value)
        if registry_name is None:
            continue
        if not node.args:
            continue
        class_name = _class_name_from_expr(node.args[0], known_classes)
        if class_name is None:
            continue
        key_source = node.args[1] if len(node.args) >= 2 else node.args[0]
        key_fingerprint = _fingerprint_builder_value(key_source)
        shapes.append(
            RegistrationShape.from_registration_call(
                parsed_module,
                node,
                registry_name,
                class_name,
                key_fingerprint,
            )
        )
    return shapes


def _collect_decorator_registration_shapes(
    parsed_module: ParsedModule,
) -> list[RegistrationShape]:
    shapes: list[RegistrationShape] = []
    for node in ast.walk(parsed_module.module):
        if not isinstance(node, ast.ClassDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            decorator_name = _call_name(decorator.func)
            if decorator_name not in {
                "register",
                "register_class",
                "register_type",
                "auto_register",
            }:
                continue
            if not decorator.args:
                continue
            registry_name = _call_name(decorator.args[0])
            if registry_name is None:
                continue
            key_expr = (
                decorator.args[1]
                if len(decorator.args) >= 2
                else ast.Constant(value=node.name)
            )
            shapes.append(
                RegistrationShape.from_decorator(
                    parsed_module,
                    node,
                    registry_name,
                    _fingerprint_builder_value(key_expr),
                )
            )
    return shapes


def _class_name_from_expr(node: ast.AST, known_classes: set[str]) -> str | None:
    if isinstance(node, ast.Name) and node.id in known_classes:
        return node.id
    return None
