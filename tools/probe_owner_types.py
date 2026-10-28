"""#92 measurement: what mypy says the unresolved foreign-absence-probe receivers are.

Scratch tooling, not part of the package. It needs a running dmypy daemon
started with ``--export-types`` over the same tree, for example::

    dmypy --status-file S run --export-types -- --strict-equality \\
        --ignore-missing-imports --python-version 3.11 nominal_refactor_advisor
    python tools/probe_owner_pairs.py nominal_refactor_advisor pairs.json
    python tools/probe_owner_types.py nominal_refactor_advisor S out.json pairs.json

For each probe whose receiver has no annotation, it asks
``dmypy inspect --show type`` at the receiver's span. For each frozen
dataclass field declared ``X | None`` that a probe reads, it asks the same about
every construction argument the static measurement could not decide.
dmypy prints short type names, so a type counts as an own class only when its
name is one scanned class's name and no other scanned class shares it.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.class_index import ModuleNominalBindingAuthority
from nominal_refactor_advisor.restated_fact_observations import (
    ForeignAbsenceSiteFamily,
    ResolvedProbeOwner,
    UnresolvedProbeOwner,
)

PACKAGE = Path(sys.argv[1])
STATUS = sys.argv[2]
OUT = Path(sys.argv[3])
STATIC = Path(sys.argv[4])  # tools/probe_owner_pairs.py output for the same tree
DMYPY = str(Path(sys.executable).with_name("dmypy"))
NO_ANNOTATION = re.compile(r"`[^`]+` has no annotation in")


@dataclass(frozen=True)
class Inspected:
    type_text: str
    seconds: float


def inspect(file_path: str, node: ast.expr) -> Inspected:
    span = f"{file_path}:{node.lineno}:{node.col_offset + 1}:{node.end_lineno}:{node.end_col_offset}"
    started = time.perf_counter()
    completed = subprocess.run(
        [DMYPY, "--status-file", STATUS, "inspect", "--show", "type", span],
        capture_output=True,
        text=True,
        check=False,
    )
    seconds = time.perf_counter() - started
    text = (completed.stdout.strip() or completed.stderr.strip()).splitlines()
    return Inspected(text[0].strip('"') if text else "<no output>", seconds)


modules = parse_python_modules(PACKAGE)
own_names: Counter[str] = Counter()
own_classes: dict[str, ast.ClassDef] = {}
for module in modules:
    for node in ast.walk(module.module):
        if isinstance(node, ast.ClassDef):
            own_names[node.name] += 1


def classify(type_text: str) -> str:
    if type_text.startswith(("<no output>", "Error", "No ")) or "error" in type_text.lower():
        return "error"
    if type_text == "Any" or "Any" == type_text.split("[")[0]:
        return "any"
    if type_text.startswith("type["):
        return "type object"
    parts = [part.strip() for part in type_text.split("|")]
    present = [part for part in parts if part != "None"]
    if len(present) != 1:
        return "union"
    name = present[0].split("[")[0].split(".")[-1]
    if name == "Any":
        return "any"
    if own_names[name] == 1:
        return "own class" + (" | None" if "None" in parts else "")
    if own_names[name] > 1:
        return "own name, ambiguous"
    return "foreign class" + (" | None" if "None" in parts else "")


# ---- step 1/2: the probes whose receiver has no annotation
receivers = []
for module in modules:
    compares: dict[tuple[int, str, str], list[ast.expr]] = defaultdict(list)
    for node in ast.walk(module.module):
        match node:
            case ast.Compare(
                left=ast.Attribute(value=receiver, attr=attribute),
                ops=[ast.Is() | ast.IsNot()],
                comparators=[ast.Constant(value=None)],
            ):
                compares[(node.lineno, ast.unparse(receiver), attribute)].append(receiver)
    for site in ForeignAbsenceSiteFamily.sites(module):
        owner = site.owner
        if not (
            isinstance(owner, UnresolvedProbeOwner) and NO_ANNOTATION.match(owner.reason)
        ):
            continue
        nodes = compares[(site.lineno, site.group, site.detail[0])]
        receiver = nodes.pop(0)
        result = inspect(module.file_path, receiver)
        receivers.append(
            {
                "site": f"{module.file_path}:{site.lineno}",
                "receiver": site.group,
                "field": site.detail[0],
                "type": result.type_text,
                "kind": classify(result.type_text),
                "seconds": round(result.seconds, 4),
            }
        )

# ---- step 3: undecided construction arguments of the frozen `X | None` pairs
measured = json.loads(STATIC.read_text())
frozen_optional = [
    pair
    for pair in measured["pairs"]
    if pair["declared_directly"] and pair["optional"] and pair["frozen"]
]
by_path = {module.file_path: module for module in modules}
arguments = []
for pair in frozen_optional:
    field_name = pair["field"]
    class_name = pair["class"].rsplit(".", 1)[1]
    declaring = next(
        node
        for module in modules
        for node in ast.walk(module.module)
        if isinstance(node, ast.ClassDef)
        and node.name == class_name
        and f"{module.module_name}." in pair["class"]
    )
    order = [
        s.target.id
        for s in declaring.body
        if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
    ]
    for location in pair["undecided_constructions"]:
        file_path, line = location.rsplit(":", 1)
        module = by_path[file_path]
        bindings = ModuleNominalBindingAuthority(module)
        for node in ast.walk(module.module):
            if not (isinstance(node, ast.Call) and node.lineno == int(line)):
                continue
            passed = [k.value for k in node.keywords if k.arg == field_name]
            if not passed and field_name in order and order.index(field_name) < len(node.args):
                passed = [node.args[order.index(field_name)]]
            for value in passed:
                result = inspect(file_path, value)
                parts = [p.strip() for p in result.type_text.split("|")]
                arguments.append(
                    {
                        "pair": f"{pair['class']}.{field_name}",
                        "site": location,
                        "argument": ast.unparse(value),
                        "type": result.type_text,
                        # any non-None type is present, a union of classes included
                        "present": "None" not in parts
                        and "Any" not in parts
                        and classify(result.type_text) not in ("any", "error"),
                        "seconds": round(result.seconds, 4),
                    }
                )

pairs_now_decidable = []
for pair in frozen_optional:
    key = pair["class"] + "." + pair["field"]
    mine = [a for a in arguments if a["pair"] == key]
    # a construction line is decided when every argument inspected there is present
    undecided_left = [
        location
        for location in pair["undecided_constructions"]
        if not (
            [a for a in mine if a["site"] == location]
            and all(a["present"] for a in mine if a["site"] == location)
        )
    ]
    if pair["constructions"] and not undecided_left and all(
        state != "undecided" for state in pair["replace_states"]
    ):
        pairs_now_decidable.append(key)

times = [r["seconds"] for r in receivers] + [a["seconds"] for a in arguments]
summary = {
    "unannotated_receivers": len(receivers),
    "receiver_kinds": dict(Counter(r["kind"] for r in receivers).most_common()),
    "frozen_optional_pairs": len(frozen_optional),
    "undecided_arguments_inspected": len(arguments),
    "arguments_present_by_type": sum(a["present"] for a in arguments),
    "pairs_decidable_with_types": len(pairs_now_decidable),
    "pairs_decidable": pairs_now_decidable,
    "inspect_calls": len(times),
    "inspect_seconds_mean": round(sum(times) / len(times), 4) if times else 0,
    "inspect_seconds_max": max(times, default=0),
}
OUT.write_text(json.dumps({"summary": summary, "receivers": receivers, "arguments": arguments}, indent=1))
print(json.dumps(summary, indent=1))
