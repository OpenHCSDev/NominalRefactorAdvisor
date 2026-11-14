"""Methods with the same name, signature and body declared in more than one class.

Usage: same_body.py PACKAGE_ROOT [MIN_OWNERS]
Trivial bodies (pass, ..., raise NotImplementedError, docstring only) are left out.
"""
import ast
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1])
MIN_OWNERS = int(sys.argv[2]) if len(sys.argv) > 2 else 2


def trivial(body):
    statements = [
        s for s in body
        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
    ]
    if not statements:
        return True
    if len(statements) == 1:
        s = statements[0]
        if isinstance(s, ast.Pass):
            return True
        if isinstance(s, ast.Raise) and "NotImplementedError" in ast.dump(s):
            return True
    return False


groups = defaultdict(list)
for path in sorted(ROOT.rglob("*.py")):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if trivial(item.body):
                continue
            body = [
                s for s in item.body
                if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
            ]
            key = (
                item.name,
                ast.dump(item.args),
                tuple(ast.dump(d) for d in item.decorator_list),
                tuple(ast.dump(s) for s in body),
            )
            lines = item.end_lineno - item.lineno + 1
            groups[key].append((str(path.relative_to(ROOT)), node.name, item.lineno, lines))

rows = [
    (len(owners), max(o[3] for o in owners), key[0], owners)
    for key, owners in groups.items()
    if len({(o[0], o[1]) for o in owners}) >= MIN_OWNERS
]
rows.sort(key=lambda r: (-r[0] * r[1], r[2]))
print(len(rows), "groups;", sum(r[0] for r in rows), "declarations")
for count, lines, name, owners in rows:
    print(f"\n{count} x {name} ({lines} lines)")
    for file, cls, lineno, _ in owners:
        print(f"    {file}:{lineno} {cls}")
