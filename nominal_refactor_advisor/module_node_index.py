"""One walk of a parsed module, shared by every pass that reads its nodes."""

from __future__ import annotations

import ast
from collections import Counter, deque
from dataclasses import dataclass, field

from .ast_tools import ParsedModule, ParsedModuleFact


@dataclass(frozen=True)
class ModuleNodeIndex(ParsedModuleFact):
    """The nodes of one module that consumer passes look up, from a single walk.

    The walk is `ast.walk`'s own breadth-first order, so every list keeps the
    order a pass that walked the module itself would have seen. It is a fact of the
    parse (`parsed.fact(ModuleNodeIndex)`), and a plan's snapshots hand unchanged
    parses from stage to stage, so each unchanged file is walked once per plan instead of once per
    pass, per operation, per stage.

    `bound_name_counts` counts each name's bindings, where every `import pkg.x`
    binds the same package object to `pkg` and so counts once however many."""

    imports: tuple[ast.Import | ast.ImportFrom, ...]
    string_constants: tuple[ast.Constant, ...]
    calls: tuple[ast.Call, ...]
    names_by_id: dict[str, tuple[ast.Name, ...]] = field(repr=False)
    parents: dict[ast.AST, ast.AST] = field(repr=False)
    loads: Counter[str] = field(repr=False)
    attribute_bases: frozenset[str]
    bound_name_counts: Counter[str] = field(repr=False)

    @classmethod
    def derive(cls, parsed_module: ParsedModule) -> ModuleNodeIndex:
        imports: list[ast.Import | ast.ImportFrom] = []
        strings: list[ast.Constant] = []
        calls: list[ast.Call] = []
        names: dict[str, list[ast.Name]] = {}
        parents: dict[ast.AST, ast.AST] = {}
        loads: Counter[str] = Counter()
        attribute_bases: set[str] = set()
        bound: Counter[str] = Counter()
        package_heads: set[str] = set()
        pending: deque[ast.AST] = deque([parsed_module.module])
        while pending:
            node = pending.popleft()
            for child in ast.iter_child_nodes(node):
                parents[child] = node
                pending.append(child)
            match node:
                case ast.Import(names=aliases):
                    imports.append(node)
                    for alias in aliases:
                        if alias.asname is None:
                            package_heads.add(alias.name.split(".")[0])
                        else:
                            bound[alias.asname] += 1
                case ast.ImportFrom(names=aliases):
                    imports.append(node)
                    bound.update(
                        alias.asname or alias.name for alias in aliases if alias.name != "*"
                    )
                case ast.arg(arg=name):
                    bound[name] += 1
                case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name) | ast.ClassDef(
                    name=name
                ):
                    bound[name] += 1
                case ast.Constant(value=str()):
                    strings.append(node)
                case ast.Call():
                    calls.append(node)
                case ast.Name(id=name, ctx=context):
                    names.setdefault(name, []).append(node)
                    if isinstance(context, ast.Load):
                        loads[name] += 1
                    else:
                        bound[name] += 1
                case ast.Attribute(value=ast.Name(id=name)):
                    attribute_bases.add(name)
        bound.update(package_heads)
        return cls(
            imports=tuple(imports),
            string_constants=tuple(strings),
            calls=tuple(calls),
            names_by_id={name: tuple(nodes) for name, nodes in names.items()},
            parents=parents,
            loads=loads,
            attribute_bases=frozenset(attribute_bases),
            bound_name_counts=bound,
        )

    def import_froms(self) -> tuple[ast.ImportFrom, ...]:
        return tuple(node for node in self.imports if isinstance(node, ast.ImportFrom))

    def names(self, name: str) -> tuple[ast.Name, ...]:
        return self.names_by_id.get(name, ())
