"""Detect `@functools.singledispatch` / `@singledispatchmethod` switching on owned classes (IMPL-3 / IMPL-4)."""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from nominal_refactor_advisor.ast_tools import ParsedModule, ParsedModuleSet
from nominal_refactor_advisor.class_families.parsed_index import (
    ClassFamilyIndex,
    ModuleClassReferenceResolver,
    build_class_family_index,
)
from nominal_refactor_advisor.detectors._base import (
    CrossModuleCandidateDetector,
    DetectorConfig,
    finding_spec_template,
)
from nominal_refactor_advisor.models import RefactorFinding, SourceLocation
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag


_DISPATCH_DECORATOR_NAMES = frozenset({"singledispatch", "singledispatchmethod"})


@dataclass(frozen=True)
class OwnedSingleDispatchCandidate:
    """One `@singledispatch` / `@singledispatchmethod` dispatcher registering owned classes."""

    file_path: str
    line: int
    dispatcher_name: str
    decorator_name: str
    owned_target_symbols: tuple[str, ...]
    registration_lines: tuple[int, ...]


def _decorator_simple_name(decorator: ast.expr) -> str | None:
    if isinstance(decorator, ast.Name):
        return decorator.id
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    return None


def _extract_registered_type_nodes(expr: ast.expr) -> tuple[ast.expr, ...]:
    """Unwrap `T`, `A | B`, `typing.Union[A, B]`, or `(A, B)` type targets."""

    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
        return (
            *_extract_registered_type_nodes(expr.left),
            *_extract_registered_type_nodes(expr.right),
        )
    if isinstance(expr, ast.Tuple):
        out: list[ast.expr] = []
        for elt in expr.elts:
            out.extend(_extract_registered_type_nodes(elt))
        return tuple(out)
    if (
        isinstance(expr, ast.Subscript)
        and _decorator_simple_name(expr.value) == "Union"
    ):
        return _extract_registered_type_nodes(expr.slice)
    if isinstance(expr, (ast.Name, ast.Attribute)):
        return (expr,)
    return ()


def _first_dispatched_parameter_annotation(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    is_method: bool,
) -> ast.expr | None:
    positional = (*node.args.posonlyargs, *node.args.args)
    index = 1 if is_method else 0
    if len(positional) <= index:
        return None
    return positional[index].annotation


def _collect_module_singledispatch_candidates(
    module: ParsedModule,
    index: ClassFamilyIndex,
) -> list[OwnedSingleDispatchCandidate]:
    if "singledispatch" not in module.source:
        return []
    resolver = ModuleClassReferenceResolver(module, index)
    dispatchers: dict[str, tuple[int, str, bool]] = {}
    registered_by_dispatcher: dict[str, list[tuple[str, int]]] = defaultdict(list)

    def _visit_block(
        statements: Sequence[ast.stmt],
        *,
        enclosing_class: str | None = None,
    ) -> None:
        for stmt in statements:
            if isinstance(stmt, ast.ClassDef):
                _visit_block(stmt.body, enclosing_class=stmt.name)
                continue
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            qualname = (
                f"{enclosing_class}.{stmt.name}"
                if enclosing_class is not None
                else stmt.name
            )
            for dec in stmt.decorator_list:
                dec_name = _decorator_simple_name(dec)
                if dec_name in _DISPATCH_DECORATOR_NAMES:
                    dispatchers[qualname] = (
                        stmt.lineno,
                        dec_name,
                        dec_name == "singledispatchmethod" or enclosing_class is not None,
                    )
                elif (
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr == "register"
                    and isinstance(dec.func.value, ast.Name)
                    and dec.args
                ):
                    target_name = (
                        f"{enclosing_class}.{dec.func.value.id}"
                        if enclosing_class is not None
                        and f"{enclosing_class}.{dec.func.value.id}" in dispatchers
                        else dec.func.value.id
                    )
                    for type_node in _extract_registered_type_nodes(dec.args[0]):
                        symbol = resolver.symbol_for_reference(type_node)
                        if symbol is not None and symbol in index.classes_by_symbol:
                            registered_by_dispatcher[target_name].append(
                                (symbol, stmt.lineno)
                            )
                elif (
                    isinstance(dec, ast.Attribute)
                    and dec.attr == "register"
                    and isinstance(dec.value, ast.Name)
                ):
                    target_name = (
                        f"{enclosing_class}.{dec.value.id}"
                        if enclosing_class is not None
                        and f"{enclosing_class}.{dec.value.id}" in dispatchers
                        else dec.value.id
                    )
                    is_method = enclosing_class is not None
                    if target_name in dispatchers:
                        is_method = dispatchers[target_name][2]
                    ann = _first_dispatched_parameter_annotation(
                        stmt, is_method=is_method
                    )
                    if ann is not None:
                        for type_node in _extract_registered_type_nodes(ann):
                            symbol = resolver.symbol_for_reference(type_node)
                            if symbol is not None and symbol in index.classes_by_symbol:
                                registered_by_dispatcher[target_name].append(
                                    (symbol, stmt.lineno)
                                )

    _visit_block(module.module.body)
    found: list[OwnedSingleDispatchCandidate] = []
    for dispatcher_name, (line, decorator_name, _is_method) in sorted(
        dispatchers.items(), key=lambda item: (item[1][0], item[0])
    ):
        regs = registered_by_dispatcher.get(dispatcher_name, ())
        if not regs:
            continue
        owned_symbols = tuple(dict.fromkeys(sym for sym, _ln in regs))
        reg_lines = tuple(dict.fromkeys(ln for _sym, ln in regs))
        found.append(
            OwnedSingleDispatchCandidate(
                file_path=module.file_path,
                line=line,
                dispatcher_name=dispatcher_name,
                decorator_name=decorator_name,
                owned_target_symbols=owned_symbols,
                registration_lines=reg_lines,
            )
        )
    return found


class OwnedSingleDispatchDetector(
    CrossModuleCandidateDetector[OwnedSingleDispatchCandidate]
):
    """Detect `@singledispatch` / `@singledispatchmethod` dispatching over owned classes."""

    finding_spec = finding_spec_template(
        PatternId.OWNED_SINGLEDISPATCH,
        "Owned classes dispatched via `@singledispatch` instead of methods (IMPL-3 / IMPL-4)",
        "Registering repository-owned classes on an external `@singledispatch` function duplicates class dispatch outside the family (the same slop as an `isinstance` chain), forcing every new subclass to add a registration in another function.",
        "a default method on the family base and overrides on the registered subclasses",
        "`@singledispatch` or `@singledispatchmethod` whose `.register(...)` targets resolve to classes declared in this repository",
        (CapabilityTag.AUTHORITATIVE_DISPATCH, CapabilityTag.MRO_ORDERING),
        (ObservationTag.BRANCH_DISPATCH, ObservationTag.CLASS_FAMILY),
    )

    def _candidate_items(
        self,
        modules: ParsedModuleSet,
        config: DetectorConfig,
    ) -> Sequence[OwnedSingleDispatchCandidate]:
        del config
        index = build_class_family_index(modules)
        candidates: list[OwnedSingleDispatchCandidate] = []
        for module in modules:
            candidates.extend(_collect_module_singledispatch_candidates(module, index))
        return tuple(
            sorted(
                candidates,
                key=lambda item: (item.file_path, item.line, item.dispatcher_name),
            )
        )

    def _finding_for_candidate(
        self, candidate: OwnedSingleDispatchCandidate
    ) -> RefactorFinding:
        targets = ", ".join(
            f"`{sym.rsplit('.', 1)[-1]}`" for sym in candidate.owned_target_symbols[:6]
        )
        extra = (
            f" (+{len(candidate.owned_target_symbols) - 6} more)"
            if len(candidate.owned_target_symbols) > 6
            else ""
        )
        return self.build_finding(
            f"`@{candidate.decorator_name}` on `{candidate.dispatcher_name}` dispatches "
            f"externally over {len(candidate.owned_target_symbols)} owned class(es) "
            f"({targets}{extra}) (IMPL-3 / IMPL-4); move the default onto the family "
            "base and each registered arm onto its owning class.",
            (
                SourceLocation(
                    candidate.file_path,
                    candidate.line,
                    candidate.dispatcher_name,
                ),
            ),
        )


__all__ = (
    "OwnedSingleDispatchCandidate",
    "OwnedSingleDispatchDetector",
)
