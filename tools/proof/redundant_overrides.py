"""Methods whose body, signature and decorators equal the one they would inherit.

Usage: redundant_overrides.py PACKAGE_ROOT
A base is resolved by simple class name: the class of that name in the same file, else the
package's only class of that name; the
inherited method is the first one found walking the bases depth-first, left to right
(an approximation of the MRO that is exact for single inheritance).
"""
import ast
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1])
classes = defaultdict(list)
for path in sorted(ROOT.rglob("*.py")):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ClassDef):
            classes[node.name].append((str(path.relative_to(ROOT)), node))


def base_names(node):
    for base in node.bases:
        target = base.value if isinstance(base, ast.Subscript) else base
        if isinstance(target, ast.Name):
            yield target.id
        elif isinstance(target, ast.Attribute):
            yield target.attr


def methods(node):
    return {
        item.name: item
        for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def key(function):
    body = [
        s for s in function.body
        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
    ]
    return (
        ast.dump(function.args),
        ast.dump(function.returns) if function.returns else "",
        tuple(ast.dump(d) for d in function.decorator_list),
        tuple(ast.dump(s) for s in body),
    )


def resolve(base, path):
    declared = classes.get(base, ())
    local = [d for d in declared if d[0] == path]
    if len(local) == 1:
        return local[0]
    return declared[0] if len(declared) == 1 else None


def inherited(node, path, name, seen):
    for base in base_names(node):
        resolved = resolve(base, path)
        if base in seen or resolved is None:
            continue
        seen.add(base)
        path, base_node = resolved
        found = methods(base_node).get(name)
        if found is not None:
            return base, found
        deeper = inherited(base_node, path, name, seen)
        if deeper is not None:
            return deeper
    return None


rows = []
for name, declared in classes.items():
    for path, node in declared:
        for method_name, function in methods(node).items():
            found = inherited(node, path, method_name, set())
            if found is None:
                continue
            owner, parent = found
            if key(function) == key(parent):
                lines = function.end_lineno - function.lineno + 1
                rows.append((lines, f"{path}:{function.lineno} {name}.{method_name} == {owner}.{method_name}"))
rows.sort(reverse=True)
print(len(rows), "overrides equal to what they inherit;", sum(r[0] for r in rows), "lines")
for lines, row in rows:
    print(f"{lines:3d}  {row}")
