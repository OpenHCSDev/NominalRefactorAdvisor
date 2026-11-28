"""Convert a dotted class-name string roster into a nominal capability mixin (MEMB-2).

A hand-typed `frozenset[str]` of `"pkg.mod.ClassName"` strings checked against
`f"{cls.__module__}.{cls.__name__}"` restates capability membership outside the class
declarations (Paper 1 Prop 4.2, Cor 4.12). Every new subclass drifts from the roster,
and `bool(cls.__subclasses__())` guards hide trainable classes behind held-out
subclasses.

This module owns both:
- the shared AST extraction and exact DAG minimal sound ancestor cover algorithm used by
  `DottedClassRosterDetector`, and
- `RosterToCapabilityMixinOperation` (`roster_to_capability_mixin`), which declares the
  capability mixin, inserts absolute imports and base-list entries onto the minimal
  sound ancestor cover across modules, deletes the string roster, and rewrites
  dotted-name membership checks to `issubclass(..., Mixin)` (or `isinstance` /
  `members_with`).
"""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from nominal_refactor_advisor.ast_tools import ParsedModule
from nominal_refactor_advisor.codemod_declaration_source import (
    ClassHeaderSpanSourceAuthority,
)
from nominal_refactor_advisor.codemod_imports import ModuleImportMutation
from nominal_refactor_advisor.codemod_module_declarations import ModuleSymbolTable
from nominal_refactor_advisor.codemod_payload import (
    OptionalStringPayloadValueCodec,
    RequiredStringPayloadValueCodec,
    codemod_payload_field,
)
from nominal_refactor_advisor.codemod_preflight import (
    CodemodOperationPreflightError,
    CodemodOperationPreflightReport,
)
from nominal_refactor_advisor.codemod_reproof import (
    RepositorySourceReprovedOperation,
    SourceReproofDiagnostic,
)
from nominal_refactor_advisor.codemod_runtime import CodemodSourceSnapshot
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_semantics import CodemodPreflightStatus
from nominal_refactor_advisor.codemod_source_edits import (
    NominalSourceEdit,
    SourceTextGeometry,
    SourceTextSpanReplacement,
)
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.refactors.class_family_collection import (
    ClassFamilyCollectionFactory,
)


class IndexedClassGraph(Protocol):
    """Minimal graph interface shared by `ClassFamilyIndex` and `CompactClassFamilyIndex`."""

    @property
    def classes_by_symbol(self) -> dict[str, object]: ...

    @property
    def children_by_symbol(self) -> dict[str, tuple[str, ...]]: ...

    def ancestor_symbols(self, symbol: str) -> tuple[str, ...]: ...

    def descendant_symbols(self, symbol: str) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class ModuleDottedRoster:
    """One module-level or class-level string collection naming dotted class paths."""

    file_path: str
    statement: ast.Assign | ast.AnnAssign
    name: str
    qualname: str
    entries: tuple[str, ...]
    line: int


def is_dotted_class_string(value: object) -> bool:
    """Whether `value` is a syntactically valid dotted Python symbol path."""

    if not isinstance(value, str) or "." not in value:
        return False
    parts = value.split(".")
    return len(parts) >= 2 and all(part.isidentifier() for part in parts)


def extract_dotted_string_roster(
    value: ast.AST,
    unavailable_builtin_names: frozenset[str] = frozenset(),
) -> tuple[str, ...] | None:
    """Extract a `>= 2` string-literal collection whose elements are dotted symbols."""

    for factory in ClassFamilyCollectionFactory:
        elements = factory.elements(value, unavailable_builtin_names)
        if elements is None or len(elements) < 2:
            continue
        strings: list[str] = []
        for element in elements:
            if not (
                isinstance(element, ast.Constant)
                and isinstance(element.value, str)
                and is_dotted_class_string(element.value)
            ):
                strings = []
                break
            strings.append(element.value)
        if len(strings) == len(elements):
            return tuple(strings)
    return None


def _assignment_name_and_value(
    statement: ast.stmt,
) -> tuple[str, ast.expr] | None:
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        return statement.targets[0].id, statement.value
    if (
        isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.value is not None
    ):
        return statement.target.id, statement.value
    return None


def extract_dotted_class_rosters(
    module: ParsedModule,
) -> tuple[ModuleDottedRoster, ...]:
    """Extract top-level and class-body dotted class-name string collections."""

    rosters: list[ModuleDottedRoster] = []
    for statement in module.module.body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            bound = _assignment_name_and_value(statement)
            if bound is None or bound[0] == "__all__":
                continue
            entries = extract_dotted_string_roster(bound[1])
            if entries is not None:
                rosters.append(
                    ModuleDottedRoster(
                        file_path=module.file_path,
                        statement=statement,
                        name=bound[0],
                        qualname=bound[0],
                        entries=entries,
                        line=statement.lineno,
                    )
                )
        elif isinstance(statement, ast.ClassDef):
            for body_stmt in statement.body:
                if not isinstance(body_stmt, (ast.Assign, ast.AnnAssign)):
                    continue
                bound = _assignment_name_and_value(body_stmt)
                if bound is None:
                    continue
                entries = extract_dotted_string_roster(bound[1])
                if entries is not None:
                    rosters.append(
                        ModuleDottedRoster(
                            file_path=module.file_path,
                            statement=body_stmt,
                            name=bound[0],
                            qualname=f"{statement.name}.{bound[0]}",
                            entries=entries,
                            line=body_stmt.lineno,
                        )
                    )
    return tuple(rosters)


def dotted_name_format_subject(expr: ast.AST) -> ast.expr | None:
    """Match `f"{x.__module__}.{x.__name__}"`, `f"{x.__module__}.{x.__qualname__}"`,
    or `x.__module__ + "." + x.__name__` and return the class expression `x`."""

    if isinstance(expr, ast.JoinedStr) and len(expr.values) == 3:
        first, middle, last = expr.values
        if (
            isinstance(first, ast.FormattedValue)
            and isinstance(middle, ast.Constant)
            and middle.value == "."
            and isinstance(last, ast.FormattedValue)
            and isinstance(first.value, ast.Attribute)
            and first.value.attr == "__module__"
            and isinstance(last.value, ast.Attribute)
            and last.value.attr in {"__name__", "__qualname__"}
            and ast.dump(first.value.value) == ast.dump(last.value.value)
        ):
            return first.value.value
    if (
        isinstance(expr, ast.BinOp)
        and isinstance(expr.op, ast.Add)
        and isinstance(expr.left, ast.BinOp)
        and isinstance(expr.left.op, ast.Add)
        and isinstance(expr.left.right, ast.Constant)
        and expr.left.right.value == "."
        and isinstance(expr.left.left, ast.Attribute)
        and expr.left.left.attr == "__module__"
        and isinstance(expr.right, ast.Attribute)
        and expr.right.attr in {"__name__", "__qualname__"}
        and ast.dump(expr.left.left.value) == ast.dump(expr.right.value)
    ):
        return expr.left.left.value
    return None


def module_compares_roster_against_dotted_names(
    tree: ast.Module,
    roster_name: str,
) -> bool:
    """Whether `tree` compares `roster_name` against dotted `__module__.__name__` strings."""

    qualname_set_vars: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            bound = _assignment_name_and_value(node)
            if bound is not None and _qualname_set_comp_source(bound[1]) is not None:
                qualname_set_vars.add(bound[0])
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
            op = node.ops[0]
            comp = node.comparators[0]
            if (
                isinstance(op, (ast.In, ast.NotIn))
                and isinstance(comp, ast.Name)
                and comp.id == roster_name
                and dotted_name_format_subject(node.left) is not None
            ):
                return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
            sides = (node.left, node.right)
            if any(isinstance(s, ast.Name) and s.id == roster_name for s in sides):
                other = sides[1] if isinstance(sides[0], ast.Name) and sides[0].id == roster_name else sides[0]
                if (
                    isinstance(other, ast.Name)
                    and other.id in qualname_set_vars
                ) or _qualname_set_comp_source(other) is not None:
                    return True
    return False


def most_specific_common_ancestor(
    class_index: IndexedClassGraph,
    symbols: Sequence[str],
) -> str | None:
    """Return the most-specific indexed ancestor shared by every symbol, if any."""

    if not symbols:
        return None
    common = set(class_index.ancestor_symbols(symbols[0]))
    for symbol in symbols[1:]:
        common &= set(class_index.ancestor_symbols(symbol))
        if not common:
            return None
    # Most specific = no other common ancestor descends from it.
    specific = [
        anc
        for anc in common
        if not any(
            other != anc and anc in class_index.ancestor_symbols(other)
            for other in common
        )
    ]
    if not specific:
        return None
    return sorted(
        specific,
        key=lambda sym: (len(class_index.descendant_symbols(sym)), sym),
    )[0]


def minimal_sound_ancestor_cover(
    class_index: IndexedClassGraph,
    target_symbols: frozenset[str],
    *,
    family_root: str | None = None,
) -> tuple[str, ...]:
    """Compute a minimum-cardinality sound ancestor cover of `target_symbols` in the DAG.

    A class `c` is a sound cover candidate iff:
    - `c in target_symbols`, OR
    - `c` covers at least two targets (`|descendants(c) & target_symbols| >= 2`),
      `c` is neither `family_root` nor an ancestor of `family_root`, AND every leaf
      descendant of `c` in `class_index` belongs to `target_symbols` (so tagging `c`
      never leaks capability membership onto any non-target concrete leaf class).

    Among equal-coverage sound candidates, the most specific class (fewest total
    descendants) wins, avoiding unnecessary upward lifts to broader mixins.
    """

    if not target_symbols:
        return ()

    family_ancestors = (
        frozenset((family_root, *class_index.ancestor_symbols(family_root)))
        if family_root is not None
        else frozenset()
    )

    def _is_leaf(sym: str) -> bool:
        return not class_index.children_by_symbol.get(sym)

    candidate_coverage: dict[str, frozenset[str]] = {}
    candidate_desc_count: dict[str, int] = {}

    # Every target itself is always a sound candidate covering itself + any target descendants.
    for target in target_symbols:
        desc = frozenset(class_index.descendant_symbols(target))
        covered = (desc | {target}) & target_symbols
        candidate_coverage[target] = covered
        candidate_desc_count[target] = len(desc)

    # Check non-target ancestors that cover >= 2 targets with zero non-target leaf descendants.
    seen_ancestors: set[str] = set()
    for target in target_symbols:
        seen_ancestors.update(class_index.ancestor_symbols(target))
    for anc in seen_ancestors:
        if anc in target_symbols or anc in family_ancestors:
            continue
        desc = frozenset(class_index.descendant_symbols(anc))
        covered = desc & target_symbols
        if len(covered) < 2:
            continue
        leaf_descendants = {d for d in desc if _is_leaf(d)}
        if not leaf_descendants or not (leaf_descendants <= target_symbols):
            continue
        candidate_coverage[anc] = covered
        candidate_desc_count[anc] = len(desc)

    # Prune dominated candidates: drop x if another y covers a strict superset,
    # or equal set with strictly fewer descendants (more specific) or lexicographic tie-break.
    candidates = sorted(
        candidate_coverage,
        key=lambda sym: (-len(candidate_coverage[sym]), candidate_desc_count[sym], sym),
    )
    undominated: list[str] = []
    for sym in candidates:
        cov = candidate_coverage[sym]
        dominated = False
        for kept in undominated:
            kept_cov = candidate_coverage[kept]
            if cov <= kept_cov:
                if cov < kept_cov:
                    dominated = True
                    break
                # Equal coverage: prefer the more specific candidate (already sorted first).
                if (candidate_desc_count[kept], kept) <= (candidate_desc_count[sym], sym):
                    dominated = True
                    break
        if not dominated:
            undominated.append(sym)

    # Exact minimum set cover: take forced unique-cover elements first, then exact search.
    chosen: list[str] = []
    uncovered = set(target_symbols)
    pool = list(undominated)

    while uncovered:
        # Forced elements: any uncovered target covered by exactly one remaining candidate.
        forced: list[str] = []
        for target in sorted(uncovered):
            coverers = [c for c in pool if target in candidate_coverage[c]]
            if len(coverers) == 1 and coverers[0] not in forced:
                forced.append(coverers[0])
        if forced:
            for sym in forced:
                chosen.append(sym)
                uncovered -= candidate_coverage[sym]
            pool = [c for c in pool if c not in forced and (candidate_coverage[c] & uncovered)]
            continue
        # Residual overlap (rare diamonds): exact branch-and-bound for small pools, greedy fallback.
        if len(pool) <= 24:
            best = _exact_min_set_cover(uncovered, pool, candidate_coverage)
            chosen.extend(best)
            uncovered.clear()
        else:
            best_sym = max(
                pool,
                key=lambda sym: (
                    len(candidate_coverage[sym] & uncovered),
                    -candidate_desc_count[sym],
                ),
            )
            chosen.append(best_sym)
            uncovered -= candidate_coverage[best_sym]
            pool = [c for c in pool if c != best_sym and (candidate_coverage[c] & uncovered)]

    return tuple(sorted(chosen))


def _exact_min_set_cover(
    uncovered: set[str],
    pool: list[str],
    coverage: dict[str, frozenset[str]],
) -> tuple[str, ...]:
    best: list[str] | None = None

    def _search(remaining: frozenset[str], chosen: list[str]) -> None:
        nonlocal best
        if best is not None and len(chosen) >= len(best):
            return
        if not remaining:
            best = list(chosen)
            return
        # Branch on the remaining target covered by the fewest candidates.
        pivot = min(
            remaining,
            key=lambda t: sum(1 for c in pool if t in coverage[c]),
        )
        options = sorted(
            (c for c in pool if pivot in coverage[c]),
            key=lambda c: (-len(coverage[c] & remaining), c),
        )
        for option in options:
            chosen.append(option)
            _search(remaining - coverage[option], chosen)
            chosen.pop()

    _search(frozenset(uncovered), [])
    return tuple(best or ())


def _qualname_set_comp_source(expr: ast.AST) -> tuple[str, ast.expr] | None:
    """Match `{f"{k.__module__}.{k.__name__}" for k in <iter>}` -> `(var_name, iter_expr)`."""

    if not isinstance(expr, ast.SetComp) or len(expr.generators) != 1:
        return None
    gen = expr.generators[0]
    if gen.ifs or gen.is_async or not isinstance(gen.target, ast.Name):
        return None
    subject = dotted_name_format_subject(expr.elt)
    if isinstance(subject, ast.Name) and subject.id == gen.target.id:
        return gen.target.id, gen.iter
    return None


def _subclass_or_instance_check(
    subject: ast.expr,
    mixin_name: str,
    *,
    negated: bool,
) -> str:
    if (
        isinstance(subject, ast.Call)
        and isinstance(subject.func, ast.Name)
        and subject.func.id == "type"
        and len(subject.args) == 1
        and not subject.keywords
    ):
        expr = f"isinstance({ast.unparse(subject.args[0])}, {mixin_name})"
    else:
        expr = f"issubclass({ast.unparse(subject)}, {mixin_name})"
    return f"not {expr}" if negated else expr


def _statement_line_offsets(
    source: str,
    statement: ast.stmt,
    *,
    consume_trailing_blank_lines: bool = False,
) -> tuple[int, int]:
    lines = source.splitlines(keepends=True)
    start_line = statement.lineno
    end_line = statement.end_lineno or statement.lineno
    if consume_trailing_blank_lines:
        while end_line < len(lines) and not lines[end_line].strip():
            end_line += 1
    start = sum(len(line) for line in lines[: start_line - 1])
    end = sum(len(line) for line in lines[:end_line])
    return start, end


def _resolve_snapshot_file_path(
    snapshot: CodemodSourceSnapshot,
    raw_path: str,
) -> str:
    if raw_path in snapshot.sources_by_file_path:
        return raw_path
    normalized = raw_path.replace("\\", "/").lstrip("./")
    matches = [
        path
        for path in snapshot.sources_by_file_path
        if path.replace("\\", "/").endswith(f"/{normalized}")
        or path.replace("\\", "/") == normalized
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise NeedsInput(
            f"mixin destination `{raw_path}` is not a parsed Python module in this snapshot",
            NeededInput.OWNER_PICK,
        )
    raise NeedsInput(
        f"mixin destination `{raw_path}` matched multiple modules: {sorted(matches)}",
        NeededInput.OWNER_PICK,
    )


def _quoted_docstring(docstring: str) -> str:
    escaped = docstring.strip().replace('"""', r"\"\"\"")
    return f'"""{escaped}"""'


@dataclass(frozen=True, kw_only=True)
class RosterToCapabilityMixinOperation(RepositorySourceReprovedOperation):
    """Replace a dotted class-name string roster with a nominal capability mixin (MEMB-2).

    Declares `mixin_name` in `mixin_file_path` (unless already declared there), computes
    the exact minimal sound ancestor cover of the roster's resolved classes in the
    inheritance DAG, adds absolute imports and base-list entries onto each cover class,
    deletes the string roster, and rewrites dotted-name membership checks to
    `issubclass(..., mixin_name)` (or `family_membership_call(mixin_name)` for
    collection reads when supplied).
    """

    mixin_file_path: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    mixin_name: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    docstring: str = codemod_payload_field(RequiredStringPayloadValueCodec())
    family_membership_call: str | None = codemod_payload_field(
        OptionalStringPayloadValueCodec(),
        default=None,
    )

    def preflight_reports(
        self, context: CodemodSelectorContext
    ) -> tuple[CodemodOperationPreflightReport, ...]:
        try:
            self.required_reproof(lambda: self.source_edits_from_snapshot(context))
        except CodemodOperationPreflightError as error:
            return (error.report,)
        return (
            CodemodOperationPreflightReport(
                operation=self.operation_key(),
                status=CodemodPreflightStatus.PASSED,
                message="Dotted class roster collapses cleanly onto a capability mixin",
                detail=SourceReproofDiagnostic(self.target, ()),
            ),
        )

    def source_edits_from_snapshot(
        self, snapshot: CodemodSourceSnapshot
    ) -> tuple[NominalSourceEdit, ...]:
        if not self.mixin_name.isidentifier():
            raise NeedsInput(
                f"`{self.mixin_name}` is not a valid Python class name",
                NeededInput.OWNER_PICK,
            )
        if not self.docstring.strip():
            raise NeedsInput(
                f"`{self.mixin_name}` needs a one-line docstring describing the capability",
                NeededInput.OWNER_PICK,
            )

        roster = self._locate_roster(snapshot)
        index = snapshot.required_class_family_index
        unresolved = tuple(
            entry for entry in roster.entries if entry not in index.classes_by_symbol
        )
        if unresolved:
            raise NeedsInput(
                f"`{roster.qualname}` names {len(unresolved)} class(es) absent from the "
                f"class index: {list(unresolved[:5])}; fix or remove stale entries first",
                NeededInput.SOURCE_CHANGE,
            )

        resolved_symbols = tuple(dict.fromkeys(roster.entries))
        family_root = most_specific_common_ancestor(index, resolved_symbols)
        cover_symbols = minimal_sound_ancestor_cover(
            index,
            frozenset(resolved_symbols),
            family_root=family_root,
        )
        cover_classes = tuple(
            index.classes_by_symbol[sym] for sym in cover_symbols
        )
        mixin_path = _resolve_snapshot_file_path(snapshot, self.mixin_file_path)
        mixin_module = snapshot.parsed_module_for_source_path(mixin_path)
        mixin_symbol = f"{mixin_module.module_name}.{self.mixin_name}"

        geometries: dict[str, SourceTextGeometry] = {}
        replacements_by_file: dict[str, list[SourceTextSpanReplacement]] = defaultdict(list)
        import_mutations: list[ModuleImportMutation] = []

        def geometry(file_path: str) -> SourceTextGeometry:
            return geometries.setdefault(
                file_path, SourceTextGeometry(snapshot.sources_by_file_path[file_path])
            )

        # 1. Declare the capability mixin in mixin_path unless already declared there.
        mixin_symbol_table = ModuleSymbolTable(
            file_path=mixin_module.file_path,
            source=mixin_module.source,
            module=mixin_module.module,
        )
        existing_bindings = mixin_symbol_table.binding_statements(self.mixin_name)
        already_declared = (
            len(existing_bindings) == 1
            and isinstance(existing_bindings[0], ast.ClassDef)
        )
        if existing_bindings and not already_declared:
            raise NeedsInput(
                f"`{self.mixin_name}` is already bound to a non-class in {mixin_path}",
                NeededInput.OWNER_PICK,
            )
        if not already_declared:
            mixin_source = (
                f"class {self.mixin_name}:\n"
                f"    {_quoted_docstring(self.docstring)}\n\n"
                f"    __slots__ = ()\n"
            )
            anchor_nodes: list[ast.ClassDef] = [
                cls.node for cls in cover_classes if cls.file_path == mixin_path
            ]
            if family_root is not None:
                root_cls = index.classes_by_symbol.get(family_root)
                if root_cls is not None and root_cls.file_path == mixin_path:
                    anchor_nodes.append(root_cls.node)
            if anchor_nodes:
                first_anchor = min(anchor_nodes, key=lambda node: node.lineno)
                first_line = min(
                    (dec.lineno for dec in first_anchor.decorator_list),
                    default=first_anchor.lineno,
                )
                lines = mixin_module.source.splitlines(keepends=True)
                offset = sum(len(line) for line in lines[: first_line - 1])
                replacements_by_file[mixin_path].append(
                    SourceTextSpanReplacement.from_offsets(
                        start_offset=offset,
                        end_offset=offset,
                        replacement_source=mixin_source + "\n\n",
                    )
                )
            else:
                src = mixin_module.source
                prefix = "\n\n" if src.endswith("\n") else "\n\n\n"
                if src.endswith("\n\n"):
                    prefix = "\n"
                end_offset = len(src)
                replacements_by_file[mixin_path].append(
                    SourceTextSpanReplacement.from_offsets(
                        start_offset=end_offset,
                        end_offset=end_offset,
                        replacement_source=prefix + mixin_source,
                    )
                )

        # 2. Tag every class in the minimal sound ancestor cover with the mixin base + absolute import.
        imported_files: set[str] = {mixin_path}
        for cover_cls in cover_classes:
            if (
                mixin_symbol in index.ancestor_symbols(cover_cls.symbol)
                or self.mixin_name
                in {
                    base.rsplit(".", 1)[-1]
                    for base in cover_cls.declared_base_names
                }
            ):
                continue
            if cover_cls.file_path not in imported_files:
                import_source = snapshot.module_import_graph.required_import_source(
                    importing_file_path=cover_cls.file_path,
                    imported_file_path=mixin_path,
                    imported_name=self.mixin_name,
                )
                import_mutations.append(
                    ModuleImportMutation.from_source(
                        file_path=cover_cls.file_path,
                        import_source=import_source,
                        rationale=f"Import capability mixin `{self.mixin_name}`.",
                    )
                )
                imported_files.add(cover_cls.file_path)
            header_authority = ClassHeaderSpanSourceAuthority(
                node=cover_cls.node,
                source=snapshot.sources_by_file_path[cover_cls.file_path],
            )
            replacements_by_file[cover_cls.file_path].append(
                header_authority.header_replacement(
                    header_authority.with_added_base(self.mixin_name)
                )
            )

        # 3. Rewrite all uses of the roster in its module and delete the roster declaration.
        roster_module = snapshot.parsed_module_for_source_path(roster.file_path)
        roster_geometry = geometry(roster.file_path)
        roster_replacements = self._rewrite_roster_module(
            roster_module,
            roster,
            roster_geometry,
        )
        replacements_by_file[roster.file_path].extend(roster_replacements)
        if roster.file_path not in imported_files:
            import_source = snapshot.module_import_graph.required_import_source(
                importing_file_path=roster.file_path,
                imported_file_path=mixin_path,
                imported_name=self.mixin_name,
            )
            import_mutations.append(
                ModuleImportMutation.from_source(
                    file_path=roster.file_path,
                    import_source=import_source,
                    rationale=f"Import capability mixin `{self.mixin_name}`.",
                )
            )
            imported_files.add(roster.file_path)

        rationale = self.rationale or (
            f"Replace dotted class roster `{roster.qualname}` ({len(resolved_symbols)} classes) "
            f"with capability mixin `{self.mixin_name}` on {len(cover_classes)} cover class(es)."
        )
        edits: list[NominalSourceEdit] = [
            geometry(file_path).nominal_edit(
                file_path=file_path,
                replacements=tuple(spans),
                rationale=rationale,
            )
            for file_path, spans in sorted(replacements_by_file.items())
            if spans
        ]
        edits.extend(import_mutations)
        return tuple(edits)

    def _locate_roster(self, snapshot: CodemodSourceSnapshot) -> ModuleDottedRoster:
        file_path = self.target.file_path
        qualname = self.target.qualname
        if self.target.target_id is not None:
            indexed_target = snapshot.source_index.target_for_id(self.target.target_id)
            if indexed_target is not None:
                file_path = indexed_target.file_path
                qualname = indexed_target.qualname
        if file_path is None:
            raise ValueError(
                "roster_to_capability_mixin requires target file_path or target_id"
            )
        resolved_path = _resolve_snapshot_file_path(snapshot, file_path)
        module = snapshot.parsed_module_for_source_path(resolved_path)
        rosters = extract_dotted_class_rosters(module)
        if qualname is not None:
            matched = tuple(
                r for r in rosters if r.qualname == qualname or r.name == qualname
            )
        elif self.target.line is not None:
            matched = tuple(r for r in rosters if r.line == self.target.line)
        else:
            matched = rosters
        if len(matched) != 1:
            raise ValueError(
                f"expected exactly one dotted class roster matching {self.target!r} in "
                f"{resolved_path}; found {len(matched)}"
            )
        return matched[0]

    def _rewrite_roster_module(
        self,
        module: ParsedModule,
        roster: ModuleDottedRoster,
        geometry: SourceTextGeometry,
    ) -> list[SourceTextSpanReplacement]:
        tree = module.module
        roster_name = roster.name
        roster_stmt = roster.statement
        parent_by_id: dict[int, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parent_by_id[id(child)] = parent

        # Track local `q = {f"{k.__module__}.{k.__name__}" for k in col}` helpers.
        qualname_set_defs: dict[str, tuple[ast.stmt, str, ast.expr]] = {}
        name_uses_by_id: dict[str, list[ast.Name]] = defaultdict(list)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                name_uses_by_id[node.id].append(node)
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                bound = _assignment_name_and_value(node)
                if bound is not None:
                    comp = _qualname_set_comp_source(bound[1])
                    if comp is not None:
                        qualname_set_defs[bound[0]] = (node, comp[0], comp[1])

        roster_stmt_node_ids = {id(n) for n in ast.walk(roster_stmt)}
        replacements: list[SourceTextSpanReplacement] = []
        handled_ref_ids: set[int] = set()
        deleted_helper_stmts: set[int] = set()

        for node in ast.walk(tree):
            if id(node) in roster_stmt_node_ids:
                continue
            if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
                op = node.ops[0]
                comp = node.comparators[0]
                if (
                    isinstance(op, (ast.In, ast.NotIn))
                    and isinstance(comp, ast.Name)
                    and comp.id == roster_name
                ):
                    subject = dotted_name_format_subject(node.left)
                    if subject is not None:
                        start, end = geometry.required_node_offsets(node)
                        replacements.append(
                            SourceTextSpanReplacement.from_offsets(
                                start_offset=start,
                                end_offset=end,
                                replacement_source=_subclass_or_instance_check(
                                    subject,
                                    self.mixin_name,
                                    negated=isinstance(op, ast.NotIn),
                                ),
                            )
                        )
                        handled_ref_ids.add(id(comp))
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
                left, right = node.left, node.right
                roster_side: ast.Name | None = None
                other_side: ast.expr | None = None
                if isinstance(left, ast.Name) and left.id == roster_name:
                    roster_side, other_side = left, right
                elif isinstance(right, ast.Name) and right.id == roster_name:
                    roster_side, other_side = right, left
                if roster_side is None or other_side is None:
                    continue
                inline_comp = _qualname_set_comp_source(other_side)
                if inline_comp is not None:
                    var_name, iter_expr = inline_comp
                    start, end = geometry.required_node_offsets(node)
                    replacements.append(
                        SourceTextSpanReplacement.from_offsets(
                            start_offset=start,
                            end_offset=end,
                            replacement_source=(
                                f"{{{var_name} for {var_name} in {ast.unparse(iter_expr)} "
                                f"if issubclass({var_name}, {self.mixin_name})}}"
                            ),
                        )
                    )
                    handled_ref_ids.add(id(roster_side))
                elif (
                    isinstance(other_side, ast.Name)
                    and other_side.id in qualname_set_defs
                    and len(name_uses_by_id[other_side.id]) == 1
                ):
                    helper_stmt, var_name, iter_expr = qualname_set_defs[other_side.id]
                    if id(helper_stmt) not in deleted_helper_stmts:
                        h_start, h_end = _statement_line_offsets(
                            geometry.source, helper_stmt
                        )
                        replacements.append(
                            SourceTextSpanReplacement.from_offsets(
                                start_offset=h_start,
                                end_offset=h_end,
                                replacement_source="",
                            )
                        )
                        deleted_helper_stmts.add(id(helper_stmt))
                    start, end = geometry.required_node_offsets(node)
                    replacements.append(
                        SourceTextSpanReplacement.from_offsets(
                            start_offset=start,
                            end_offset=end,
                            replacement_source=(
                                f"{{{var_name} for {var_name} in {ast.unparse(iter_expr)} "
                                f"if issubclass({var_name}, {self.mixin_name})}}"
                            ),
                        )
                    )
                    handled_ref_ids.add(id(roster_side))

        # Handle remaining references if `family_membership_call` was explicitly supplied;
        # otherwise fail loud rather than leaving a NameError.
        for ref in name_uses_by_id.get(roster_name, ()):
            if id(ref) in roster_stmt_node_ids or id(ref) in handled_ref_ids:
                continue
            if self.family_membership_call is not None:
                start, end = geometry.required_node_offsets(ref)
                replacements.append(
                    SourceTextSpanReplacement.from_offsets(
                        start_offset=start,
                        end_offset=end,
                        replacement_source=f"{self.family_membership_call}({self.mixin_name})",
                    )
                )
                handled_ref_ids.add(id(ref))
            else:
                raise NeedsInput(
                    f"`{roster.qualname}` has an unhandled reference at "
                    f"{roster.file_path}:{ref.lineno}; supply `family_membership_call` "
                    "or rewrite that site to `issubclass(..., Mixin)`",
                    NeededInput.SOURCE_CHANGE,
                )

        # Delete the string roster assignment itself.
        r_start, r_end = _statement_line_offsets(
            geometry.source,
            roster_stmt,
            consume_trailing_blank_lines=True,
        )
        replacements.append(
            SourceTextSpanReplacement.from_offsets(
                start_offset=r_start,
                end_offset=r_end,
                replacement_source="",
            )
        )
        return replacements


__all__ = (
    "IndexedClassGraph",
    "ModuleDottedRoster",
    "RosterToCapabilityMixinOperation",
    "dotted_name_format_subject",
    "extract_dotted_class_rosters",
    "extract_dotted_string_roster",
    "is_dotted_class_string",
    "minimal_sound_ancestor_cover",
    "module_compares_roster_against_dotted_names",
    "most_specific_common_ancestor",
)
