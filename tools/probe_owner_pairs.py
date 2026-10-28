"""#92 measurement: foreign absence probes whose owner is an own class field, and
whether every construction of that class decides the field's None-ness statically.

Usage: python measure.py <package dir> <out.json>
"""

from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.class_index import ModuleNominalBindingAuthority
from nominal_refactor_advisor.restated_fact_observations import (
    ForeignAbsenceSiteFamily,
    ResolvedProbeOwner,
    UnresolvedProbeOwner,
)

PACKAGE = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OPTIONAL = {"Optional", "typing.Optional"}


def is_optional(annotation: ast.expr) -> bool:
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        annotation = ast.parse(annotation.value, mode="eval").body
    match annotation:
        case ast.BinOp(op=ast.BitOr(), left=left, right=right):
            return any(
                isinstance(side, ast.Constant) and side.value is None
                for side in (left, right)
            ) or is_optional(left) or is_optional(right)
        case ast.Subscript(value=wrapper) if ast.unparse(wrapper) in OPTIONAL:
            return True
    return False


@dataclass
class OwnClass:
    qualified: str
    file_path: str
    node: ast.ClassDef
    is_dataclass: bool
    frozen: bool
    fields: dict[str, ast.AnnAssign]
    field_order: list[str]


def dataclass_flags(node: ast.ClassDef) -> tuple[bool, bool]:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if ast.unparse(target).split(".")[-1] == "dataclass":
            frozen = isinstance(decorator, ast.Call) and any(
                k.arg == "frozen"
                and isinstance(k.value, ast.Constant)
                and k.value.value is True
                for k in decorator.keywords
            )
            return True, frozen
    return False, False


modules = parse_python_modules(PACKAGE)
classes: dict[str, OwnClass] = {}
for module in modules:
    def visit(body, prefix):
        for statement in body:
            if isinstance(statement, ast.ClassDef):
                qualified = f"{module.module_name}.{prefix}{statement.name}"
                fields = {
                    s.target.id: s
                    for s in statement.body
                    if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
                }
                is_dc, frozen = dataclass_flags(statement)
                classes[qualified] = OwnClass(
                    qualified, module.file_path, statement, is_dc, frozen, fields,
                    list(fields),
                )
                visit(statement.body, f"{prefix}{statement.name}.")
    visit(module.module.body, "")

sites = [
    site
    for module in modules
    for site in ForeignAbsenceSiteFamily.sites(module)
]

# ---- construction sites per class, resolved through each module's bindings


@dataclass
class Construction:
    file_path: str
    line: int
    kind: str  # "call" | "replace"
    node: ast.Call


constructions: dict[str, list[Construction]] = defaultdict(list)
mutations: dict[tuple[str, str], list[str]] = defaultdict(list)
for module in modules:
    bindings = ModuleNominalBindingAuthority(module)
    class_stack: list[str] = []
    for node in ast.walk(module.module):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name | ast.Attribute):
            try:
                qualified = bindings.qualified_name_at(func, line=None)
            except Exception:  # measurement only: an unresolvable spelling
                qualified = None
            if qualified in classes:
                constructions[qualified].append(
                    Construction(module.file_path, node.lineno, "call", node)
                )
            if ast.unparse(func).split(".")[-1] == "replace" and node.args:
                constructions["<replace>"].append(
                    Construction(module.file_path, node.lineno, "replace", node)
                )
            if (
                ast.unparse(func) == "object.__setattr__"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
            ):
                mutations[("<any>", node.args[1].value)].append(
                    f"{module.file_path}:{node.lineno}"
                )
    # cls(...) inside a class's own classmethods constructs that class
    for qualified, own in classes.items():
        if own.file_path != module.file_path:
            continue
        for method in own.node.body:
            if isinstance(method, ast.FunctionDef) and any(
                ast.unparse(d) == "classmethod" for d in method.decorator_list
            ):
                for inner in ast.walk(method):
                    if (
                        isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Name)
                        and inner.func.id == method.args.args[0].arg
                    ):
                        constructions[qualified].append(
                            Construction(module.file_path, inner.lineno, "call", inner)
                        )
    for node in ast.walk(module.module):
        if isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute):
                    mutations[("<attr>", target.attr)].append(
                        f"{module.file_path}:{node.lineno}"
                    )


def value_state(value: ast.expr) -> str:
    """"absent", "present", or "undecided" for one supplied field value."""

    match value:
        case ast.Constant(value=None):
            return "absent"
        case ast.Constant() | ast.JoinedStr() | ast.List() | ast.Tuple() | ast.Dict() | ast.Set():
            return "present"
        case ast.ListComp() | ast.SetComp() | ast.DictComp() | ast.GeneratorExp() | ast.Lambda():
            return "present"
        case ast.Call(func=func) if ast.unparse(func).split(".")[-1][:1].isupper():
            return "present"  # a class construction
        case ast.BinOp() | ast.UnaryOp():
            return "present"
    return "undecided"


def construction_state(own: OwnClass, field_name: str, call: ast.Call) -> str:
    if any(k.arg is None for k in call.keywords) or any(
        isinstance(a, ast.Starred) for a in call.args
    ):
        return "undecided"
    for keyword in call.keywords:
        if keyword.arg == field_name:
            return value_state(keyword.value)
    if field_name in own.field_order:
        position = own.field_order.index(field_name)
        if position < len(call.args):
            return value_state(call.args[position])
    default = own.fields[field_name].value
    if default is None:
        return "undecided"  # required field missing: a subclass/inherited layout
    match default:
        case ast.Call(func=func) if ast.unparse(func).split(".")[-1] == "field":
            for k in default.keywords:
                if k.arg == "default":
                    return value_state(k.value)
                if k.arg == "default_factory":
                    return "present"
            return "undecided"
    return value_state(default)


pairs: dict[tuple[str, str], list] = defaultdict(list)
unresolved = defaultdict(int)
foreign_owner = defaultdict(int)
for site in sites:
    owner = site.owner
    if isinstance(owner, UnresolvedProbeOwner):
        unresolved[owner.reason.split("`")[0] + "`…" + owner.reason.split("`")[-1]] += 1
        continue
    assert isinstance(owner, ResolvedProbeOwner)
    if owner.qualified_name not in classes:
        foreign_owner[owner.qualified_name] += 1
        continue
    pairs[(owner.qualified_name, site.detail[0])].append(site)

report = []
for (qualified, field_name), probe_sites in sorted(pairs.items()):
    own = classes[qualified]
    declared = field_name in own.fields
    optional = declared and is_optional(own.fields[field_name].annotation)
    calls = constructions.get(qualified, [])
    replaces = [
        c
        for c in constructions.get("<replace>", [])
        if any(k.arg == field_name for k in c.node.keywords)
    ]
    states = (
        [construction_state(own, field_name, c.node) for c in calls]
        if declared
        else []
    )
    replace_states = [
        value_state(next(k.value for k in c.node.keywords if k.arg == field_name))
        for c in replaces
    ]
    mutated = mutations.get(("<attr>", field_name), []) + mutations.get(
        ("<any>", field_name), []
    )
    decidable = (
        declared
        and optional
        and own.is_dataclass
        and own.frozen
        and bool(calls)
        and all(state != "undecided" for state in states + replace_states)
    )
    report.append(
        {
            "class": qualified,
            "field": field_name,
            "declared_directly": declared,
            "annotation": ast.unparse(own.fields[field_name].annotation) if declared else None,
            "optional": optional,
            "dataclass": own.is_dataclass,
            "frozen": own.frozen,
            "constructions": len(calls),
            "construction_states": {s: states.count(s) for s in set(states)},
            "replace_sites": len(replaces),
            "replace_states": {s: replace_states.count(s) for s in set(replace_states)},
            "undecided_constructions": [
                f"{c.file_path}:{c.line}"
                for c, s in zip(calls, states)
                if s == "undecided"
            ],
            "attribute_writes_named_like_field": len(mutated),
            "decidable": decidable,
            "probe_sites": [
                f"{s.file_path}:{s.lineno} {s.group}.{s.detail[0]} {s.detail[1]}"
                for s in probe_sites
            ],
        }
    )

summary = {
    "probe_sites": len(sites),
    "owner_resolved_own_class": sum(len(v) for v in pairs.values()),
    "owner_resolved_foreign_class": sum(foreign_owner.values()),
    "owner_unresolved": sum(unresolved.values()),
    "unresolved_reasons": dict(sorted(unresolved.items(), key=lambda kv: -kv[1])),
    "foreign_owners": dict(sorted(foreign_owner.items(), key=lambda kv: -kv[1])[:20]),
    "pairs": len(report),
    "pairs_declared_optional_on_frozen_dataclass": sum(
        r["declared_directly"] and r["optional"] and r["frozen"] for r in report
    ),
    "pairs_decidable": sum(r["decidable"] for r in report),
}
OUT.write_text(json.dumps({"summary": summary, "pairs": report}, indent=1))
print(json.dumps(summary, indent=1))
