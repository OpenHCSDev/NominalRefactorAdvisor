"""Executable recipe that erases a path marked as the old one."""

from __future__ import annotations

import ast
from abc import ABC
from pathlib import Path

from nominal_refactor_advisor.ast_tools import (
    PythonSourcePathDiscovery,
    PythonSourcePathPolicy,
)
from nominal_refactor_advisor.cache_checkout import CheckoutRelocation
from nominal_refactor_advisor.declared_root import DeclaredRoot
from nominal_refactor_advisor.refactors.finding_recipes import (
    FindingRecipeSynthesizer,
    PrimaryEvidenceActionKeysMixin,
)
from nominal_refactor_advisor.refactors.target_deletion import (
    EraseDeadCompatibilityOperation,
)
from nominal_refactor_advisor.codemod_runtime import (
    CodemodSourceSnapshot,
    FindingRecipeEvaluation,
    RefactorRecipe,
)
from nominal_refactor_advisor.codemod_selection_context import CodemodSelectorContext
from nominal_refactor_advisor.codemod_selector_models import SourceRewriteTarget
from nominal_refactor_advisor.codemod_target_selectors import SourceIndexTargetSelector
from nominal_refactor_advisor.models import RefactorFinding
from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput
from nominal_refactor_advisor.refactor_concepts import LegacyPathErasureConcept
from nominal_refactor_advisor.source_index import AstTargetNodeKind


class EnclosingProjectConsumers:
    """Consumers of a legacy declaration in sibling Python files outside the scanned
    roots (such as `tools/` or `tests/` when only `src/` was scanned)."""

    @classmethod
    def outside_scan(
        cls,
        snapshot: CodemodSourceSnapshot,
        target_file_path: str,
        name: str,
    ) -> tuple[tuple[str, int], ...]:
        target_path = Path(target_file_path)
        if not target_path.is_file():
            return ()
        root = cls._project_root(target_path)
        if root is None:
            return ()
        scanned_paths = cls._scanned_resolved_paths(snapshot)
        candidate_paths = snapshot.retained_operation_derivation(
            (cls, str(root)),
            lambda: PythonSourcePathDiscovery(
                root,
                PythonSourcePathPolicy(include_tests=True),
            ).paths(),
        )
        found: list[tuple[str, int]] = []
        for path in candidate_paths:
            resolved = path.resolve()
            if resolved in scanned_paths:
                continue
            try:
                source = resolved.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if name not in source:
                continue
            line = cls._first_reference_line(source, name)
            if line is not None:
                found.append((str(resolved), line))
        return tuple(found)

    @staticmethod
    def _project_root(target_path: Path) -> Path | None:
        pyproject = DeclaredRoot.nearest_pyproject(target_path)
        if pyproject is not None:
            return pyproject.parent.resolve()
        checkout = CheckoutRelocation.containing(target_path)
        if checkout is not None:
            return checkout.root.resolve()
        return None

    @classmethod
    def _scanned_resolved_paths(
        cls, snapshot: CodemodSourceSnapshot
    ) -> frozenset[Path]:
        return snapshot.retained_operation_derivation(
            (cls, "scanned_resolved_paths"),
            lambda: frozenset(
                Path(file_path).resolve()
                for file_path in snapshot.sources_by_file_path
            ),
        )

    @staticmethod
    def _first_reference_line(source: str, name: str) -> int | None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        earliest: int | None = None
        for node in ast.walk(tree):
            line: int | None = None
            match node:
                case ast.Name(id=found, lineno=lineno) if found == name:
                    line = lineno
                case ast.Attribute(attr=found, lineno=lineno) if found == name:
                    line = lineno
                case ast.ImportFrom(names=aliases, lineno=lineno) if any(
                    alias.name == name for alias in aliases
                ):
                    line = lineno
                case ast.keyword(arg=found, value=value) if found == name:
                    line = value.lineno
            if line is not None and (earliest is None or line < earliest):
                earliest = line
        return earliest


class LegacyPathErasureFindingRecipeSynthesizer(
    PrimaryEvidenceActionKeysMixin,
    FindingRecipeSynthesizer,
    LegacyPathErasureConcept,
    ABC,
):
    """Erase the function or class a finding marks, and forbid residual calls to it.

    Only a declaration the finding's primary witness names exactly can be erased:
    a marked parameter, assignment, import or warning has no target of its own,
    so it is rejected rather than widened to its enclosing declaration. The
    operation's guard rejects the plan while any consumer still calls the path.
    """

    def evaluate_recipe_for_finding(
        self,
        finding: RefactorFinding,
        context: CodemodSelectorContext | None = None,
    ) -> FindingRecipeEvaluation:
        evidence = finding.primary_evidence
        if context is None or evidence is None:
            return self.rejected_evaluation(
                "legacy path erasure requires source context and a primary witness"
            )
        targets = [
            context.source_index.target_by_id[target_id]
            for target_id in SourceIndexTargetSelector(
                node_kinds=(
                    AstTargetNodeKind.FUNCTION,
                    AstTargetNodeKind.METHOD,
                    AstTargetNodeKind.CLASS,
                ),
                file_paths=(evidence.file_path,),
                qualnames=(evidence.symbol,),
            ).target_ids(context)
        ]
        targets = [target for target in targets if target.line == evidence.line]
        if len(targets) != 1:
            return self.rejected_evaluation(
                f"`{evidence.symbol}` at line {evidence.line} is not a function or "
                "class declaration, so only a reader can retire it"
            )
        (target,) = targets
        snapshot = context.execution_snapshot()
        consumers = snapshot.reference_index.consumers_of(
            target.name, (target.file_path, target.line)
        )
        if consumers:
            where = ", ".join(f"{path}:{line}" for path, line in consumers[:4])
            return self.stopped_evaluation(
                NeedsInput(
                    f"`{target.qualname}` still has {len(consumers)} consumer(s) "
                    f"({where}); migrate them onto the replacement before erasing it",
                    NeededInput.SOURCE_CHANGE,
                )
            )
        outside = EnclosingProjectConsumers.outside_scan(
            snapshot, target.file_path, target.name
        )
        if outside:
            where = ", ".join(f"{path}:{line}" for path, line in outside[:4])
            return self.stopped_evaluation(
                NeedsInput(
                    f"`{target.qualname}` still has {len(outside)} consumer(s) outside "
                    f"the scanned path ({where}); migrate them or scan the repository "
                    "root before erasing it",
                    NeededInput.SOURCE_CHANGE,
                )
            )
        operation = EraseDeadCompatibilityOperation(
            target=SourceRewriteTarget(target_id=target.target_id),
            forbidden_call_names=(target.name,),
            rationale="Delete the path that declares itself legacy.",
        )
        try:
            operation.source_edits(context)
        except ValueError as error:
            return self.stopped_evaluation(error)
        return self.executable_evaluation(
            RefactorRecipe(
                recipe_id=f"{finding.stable_id}-erase-legacy-path",
                reason="Erase the legacy path once its consumers use the replacement.",
            ).with_operation(operation)
        )
