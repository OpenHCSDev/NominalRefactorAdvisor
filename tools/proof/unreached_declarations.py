"""Declarations nothing alive names and no test executes, as one erase plan.

Usage: python tools/proof/unreached_declarations.py COVERAGE.json PLAN_OUT.json [READER_ROOT ...]

Run from the repository root.  COVERAGE.json is the full suite's coverage in JSON
(pytest --cov=nominal_refactor_advisor --cov-report=json:COVERAGE.json).  READER_ROOTs are
other trees that import this package; a name written there keeps a declaration.

A declaration is alive when the suite executed a line of it, when a rule below says a name
check and a test run cannot speak for it, or when its name is written anywhere outside the
span of a declaration that is itself unreached: code, a test, a doc, a string.  The result
is the largest set closed under that, so mutual callers fall together.  The plan uses
erase_dead_compatibility, whose own guard then proves no use of each name remains; on
2026-11-14 that guard caught a caller this script had first left out of the closure.

Alive by rule: handlers reached by dispatch, report properties reached by reflection,
classes with a base other than ABC/Generic (something may enumerate the family), module
bindings, dunders and visitor hooks, and the theory modules, which are Tristan's call.
"""
import ast, io, json, re, sys, tokenize
from collections import defaultdict
from pathlib import Path

ROOT = Path(".")
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".mypy_cache", ".ruff_cache", "build", "dist"}
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
extra_roots = [Path(p) for p in sys.argv[3:]]

def files(root):
    for path in root.rglob("*"):
        if path.is_file() and not (set(path.parts) & SKIP_DIRS) and path.stat().st_size < 3_000_000:
            yield path

occurrences = defaultdict(list)   # name -> [(file, line)]
for root in (ROOT, *extra_roots):
    for path in files(root):
        key = str(path)
        if path.suffix == ".py":
            try:
                source = path.read_text()
                for token in tokenize.generate_tokens(io.StringIO(source).readline):
                    if token.type == tokenize.NAME:
                        occurrences[token.string].append((key, token.start[0]))
                    elif token.type in (tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", -1)):
                        for word in set(WORD.findall(token.string)):
                            occurrences[word].append((key, token.start[0]))
            except Exception:
                pass
        elif path.suffix in {".md", ".rst", ".json", ".toml", ".yaml", ".yml", ".cfg", ".txt", ".ini", ".sh"}:
            try:
                for word in set(WORD.findall(path.read_text())):
                    occurrences[word].append((key, 0))
            except Exception:
                pass

package_classes = set()
trees = {}
for path in sorted(Path("nominal_refactor_advisor").rglob("*.py")):
    trees[str(path)] = ast.parse(path.read_text())
    package_classes |= {n.name for n in ast.walk(trees[str(path)]) if isinstance(n, ast.ClassDef)}

# Classes with a base are candidates only when nothing registers them on creation:
# an ordinary metaclass and no __init_subclass__ hook anywhere above them.
import abc, importlib, typing
def registration_free(cls):
    if type(cls) not in (type, abc.ABCMeta):
        return False
    for ancestor in cls.__mro__[1:]:
        if ancestor in (object, abc.ABC, typing.Generic) or ancestor.__module__ in ("typing", "abc", "builtins"):
            continue
        if type(ancestor) not in (type, abc.ABCMeta) or "__init_subclass__" in vars(ancestor):
            return False
        if not ancestor.__module__.startswith("nominal_refactor_advisor"):
            return False
    return "__init_subclass__" not in vars(cls)
plain_classes = set()
for file in trees:
    try:
        module = importlib.import_module(".".join(Path(file).with_suffix("").parts))
    except Exception:
        continue
    for name, value in vars(module).items():
        if isinstance(value, type) and value.__module__ == module.__name__ and registration_free(value):
            plain_classes.add((file, name))

HOOK = re.compile(r"^(__.*__|visit_.*|leave_.*|generic_visit|default)$")
defs = []   # (file, name, start, end, qualname, kind)
for file, tree in trees.items():
    def start(node):
        return min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", ()))])
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.append((file, node.name, start(node), node.end_lineno, node.name, "function"))
        elif isinstance(node, ast.ClassDef):
            bases = [ast.unparse(b) for b in node.bases]
            if (not node.bases and not node.keywords) or (file, node.name) in plain_classes:
                defs.append((file, node.name, start(node), node.end_lineno, node.name, "class"))
            foreign = any(b.split("[")[0].split(".")[-1] not in package_classes and b.split("[")[0] not in {"ABC", "Generic", "Protocol", "object"} for b in bases)
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and not HOOK.match(member.name) and not foreign:
                    defs.append((file, member.name, start(member), member.end_lineno, f"{node.name}.{member.name}", "method"))
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = node.value
            plain = isinstance(value, (ast.Name, ast.Attribute)) or (isinstance(value, ast.Call) and not value.args and not value.keywords and isinstance(value.func, ast.Name))
            if plain and node.targets[0].id != "__all__":
                defs.append((file, node.targets[0].id, node.lineno, node.end_lineno, node.targets[0].id, "binding"))

alive = set()
candidates = set(range(len(defs)))

# A declaration the full suite executed is alive whatever names it: reflection, family
# enumeration and dispatch reach code no written name points at.  Seeding the fixpoint with
# them keeps everything they name alive too.
_cov = json.loads(Path(sys.argv[1]).read_text())["files"]
def _executed(d):
    file, name, a, b, qual, kind = d
    measured = _cov.get(file)
    if measured is None:
        return True
    if kind in ("function", "method"):
        region = measured["functions"].get(qual)
        return region is None or bool(region["executed_lines"])
    if kind == "class":
        regions = [r for q, r in measured["functions"].items() if q.startswith(qual + ".")]
        return not regions or any(r["executed_lines"] for r in regions)
    return True
# What a name check and a test run cannot speak for is alive as well, and so is all it names:
# handlers reached by dispatch, report properties reached by reflection, members of a family
# something may enumerate, module bindings, and the theory modules (left to Tristan).
_THEORY = {"semantic_algebra.py", "semantic_description_length.py", "factorization.py",
           "observation_graph.py", "semantic_shape_algebra.py", "models.py"}
_DYNAMIC = ("register", "singledispatch", "json_report_property", "handles", "overload", "abstractmethod")
_PLAIN_BASES = {"ABC", "Generic", "object", "Protocol"}
def _node(file, qual):
    body, node = trees[file].body, None
    for part in qual.split("."):
        node = next((n for n in body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == part), None)
        if node is None:
            return None
        body = node.body
    return node
def _kept_by_rule(d):
    file, name, a, b, qual, kind = d
    if kind == "binding" or Path(file).name in _THEORY:
        return True
    node = _node(file, qual)
    if node is None:
        return True
    if any(word in ast.unparse(decorator) for decorator in node.decorator_list for word in _DYNAMIC):
        return True
    if kind == "class":
        bases = {ast.unparse(base).split("[")[0].split(".")[-1] for base in node.bases}
        return bool(bases - _PLAIN_BASES)
    return False
candidates = {i for i in candidates if not _executed(defs[i]) and not _kept_by_rule(defs[i])}
spans = defaultdict(list)
def rebuild():
    spans.clear()
    for i in candidates:
        spans[defs[i][0]].append((defs[i][2], defs[i][3]))
def covered(file, line):
    return any(a <= line <= b for a, b in spans.get(file, ()))
changed = True
while changed:
    rebuild()
    changed = False
    for i in sorted(candidates):
        if any(not covered(file, line) for file, line in occurrences[defs[i][1]]):
            candidates.discard(i)
            changed = True
dead = [defs[i] for i in sorted(candidates)]
# drop members whose whole class is already in the set
class_spans = [(d[0], d[2], d[3]) for d in dead if d[5] == "class"]
top = [d for d in dead if d[5] == "class" or not any(f == d[0] and a <= d[2] and d[3] <= b for f, a, b in class_spans)]
REASON = (
    "Nothing alive names this declaration (package, tests, docs, skills and the reader "
    "roots were read) and the full suite never executed a line of it."
)
operations = []
for file, name, a, b, qual, kind in top:
    operation = {
        "operation": "erase_dead_compatibility",
        "file_path": file,
        "target_qualname": qual,
        "forbidden_call_names": [name],
        "rationale": REASON,
    }
    if kind == "method":
        operation["forbidden_attribute_names"] = [name]
    operations.append(operation)
Path(sys.argv[2]).write_text(
    json.dumps({"recipes": [{"recipe_id": "erase-unreached-declarations", "reason": REASON, "operations": operations}]}, indent=1)
)
by_file = defaultdict(lambda: [0, 0])
for file, name, a, b, qual, kind in top:
    by_file[file][0] += 1; by_file[file][1] += b - a + 1
print("unnamed by anything alive and never executed:", len(top), "| lines:", sum(v[1] for v in by_file.values()))
for file, (count, lines) in sorted(by_file.items(), key=lambda kv: -kv[1][1])[:30]:
    print(f"  {lines:>5} lines {count:>4} decls  {file}")
