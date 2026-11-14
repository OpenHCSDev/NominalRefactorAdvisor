"""Every enum member in the package with every value it carries, as JSON.

Usage: PYTHONPATH=REPO python enum_dump.py OUT.json

A member's carried values are its instance attributes and the properties its class
declares.  A leading underscore is dropped from the key, so a private attribute read
through a property and a public field of the same name are one entry (and must agree).
Functions are compared by their code, not their address or line.
"""
from __future__ import annotations

import enum
import importlib
import json
import pkgutil
import sys
import types

import nominal_refactor_advisor


def code_signature(code: types.CodeType) -> list:
    return [
        code.co_code.hex(),
        list(code.co_names),
        list(code.co_varnames),
        [
            code_signature(c) if isinstance(c, types.CodeType) else repr(c)
            for c in code.co_consts
        ],
    ]


def signature(value: object) -> object:
    if isinstance(value, (types.FunctionType, types.LambdaType)):
        return ["function", code_signature(value.__code__)]
    if isinstance(value, (staticmethod, classmethod)):
        return [type(value).__name__, signature(value.__func__)]
    if isinstance(value, types.MethodType):
        return ["method", signature(value.__func__), repr(type(value.__self__).__name__)]
    if isinstance(value, type):
        return ["class", f"{value.__module__}.{value.__qualname__}"]
    if isinstance(value, (tuple, list, frozenset, set)):
        items = [signature(v) for v in value]
        if isinstance(value, (frozenset, set)):
            items = sorted(items, key=repr)
        return [type(value).__name__, items]
    if isinstance(value, dict):
        return ["dict", [[signature(k), signature(v)] for k, v in value.items()]]
    text = repr(value)
    if " at 0x" in text:
        return ["object", f"{type(value).__module__}.{type(value).__qualname__}"]
    return text


modules = []
failed = []
for info in pkgutil.walk_packages(nominal_refactor_advisor.__path__, "nominal_refactor_advisor."):
    if info.name.endswith("__main__"):
        continue
    try:
        modules.append(importlib.import_module(info.name))
    except Exception as error:  # reported, and the same on both sides of a comparison
        failed.append([info.name, f"{type(error).__name__}: {error}"[:200]])

rows = {}
for module in modules:
    for name, value in vars(module).items():
        if not (isinstance(value, type) and issubclass(value, enum.Enum)):
            continue
        if value.__module__ != module.__name__:
            continue
        properties = sorted(
            {
                attr
                for owner in value.__mro__
                if owner.__module__.startswith("nominal_refactor_advisor")
                for attr, declared in vars(owner).items()
                if isinstance(declared, property)
            }
        )
        members = []
        for member in value:
            carried = {}
            for attr, held in vars(member).items():
                if attr in ("_value_", "_name_", "__objclass__", "_sort_order_"):
                    continue
                carried[attr.lstrip("_")] = signature(held)
            for attr in properties:
                try:
                    read = signature(getattr(member, attr))
                except Exception as error:
                    read = ["raises", type(error).__name__]
                key = attr.lstrip("_")
                if key in carried and carried[key] != read:
                    carried[key + " (property)"] = read
                else:
                    carried[key] = read
            members.append([member.name, repr(member.value), dict(sorted(carried.items()))])
        rows[f"{module.__name__}.{value.__qualname__}"] = {
            "bases": [b.__name__ for b in value.__mro__[1:] if b.__module__ == "enum"],
            "members": members,
        }

with open(sys.argv[1], "w") as handle:
    json.dump({"enums": dict(sorted(rows.items())), "failed": failed}, handle, indent=1)
print(len(rows), "enums;", sum(len(r["members"]) for r in rows.values()), "members;", len(failed), "modules failed to import")
