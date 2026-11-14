"""Functions whose bodies are the same once names, attributes and constants are blanked.

Usage: same_shape.py PACKAGE_ROOT [MIN_OWNERS] [MIN_LINES]
The same procedure written again with other names: a template whose slots are the names.
"""
import ast
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1])
MIN_OWNERS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MIN_LINES = int(sys.argv[3]) if len(sys.argv) > 3 else 8


class Blank(ast.NodeTransformer):
    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id="_", ctx=node.ctx), node)

    def visit_Attribute(self, node):
        self.generic_visit(node)
        node.attr = "_"
        return node

    def visit_Constant(self, node):
        return ast.copy_location(ast.Constant(value=type(node.value).__name__), node)

    def visit_arg(self, node):
        node.arg = "_"
        node.annotation = None
        return node

    def visit_keyword(self, node):
        self.generic_visit(node)
        node.arg = "_" if node.arg else None
        return node


groups = defaultdict(list)
for path in sorted(ROOT.rglob("*.py")):
    tree = ast.parse(path.read_text())
    owners = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                owners[id(item)] = node.name
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        lines = node.end_lineno - node.lineno + 1
        if lines < MIN_LINES:
            continue
        body = [
            s for s in node.body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
        ]
        import copy
        blanked = [Blank().visit(copy.deepcopy(s)) for s in body]
        key = (len(node.args.args), tuple(ast.dump(s) for s in blanked))
        groups[key].append(
            (str(path.relative_to(ROOT)), node.lineno, owners.get(id(node), ""), node.name, lines)
        )

rows = [(len(v), max(o[4] for o in v), v) for v in groups.values() if len(v) >= MIN_OWNERS]
rows.sort(key=lambda r: -r[0] * r[1])
print(len(rows), "shapes;", sum(r[0] for r in rows), "functions;", sum(r[0] * r[1] for r in rows), "lines")
for count, lines, where in rows:
    print(f"\n{count} x {lines} lines")
    for rel, line, cls, name, _ in where:
        print(f"    {rel}:{line} {cls + '.' if cls else ''}{name}")
