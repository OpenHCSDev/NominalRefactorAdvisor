"""Extract the historical relocations of #82 from git into test fixtures.

Each fixture is source checked out from the commit that introduced a relocated
probe: the declarations and the consumer statements that branch on them, as
whole top-level definitions. Rerun to regenerate; the tests read only the files.

    python tools/relocation_fixtures.py
"""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "relocations"


@dataclass(frozen=True)
class Excerpt:
    """Top-level definitions of one file at one commit, by name or by a line they contain."""

    commit: str
    path: str
    names: tuple[str, ...] = ()
    lines: tuple[int, ...] = ()

    def source(self) -> str:
        text = subprocess.run(
            ["git", "show", f"{self.commit}:{self.path}"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout
        tree = ast.parse(text)
        chosen = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and (
                node.name in self.names
                or any(node.lineno <= line <= (node.end_lineno or node.lineno) for line in self.lines)
            )
        ]
        found = {node.name for node in chosen}
        missing = set(self.names) - found
        if missing:
            raise SystemExit(f"{self.commit}:{self.path} has no top-level {sorted(missing)}")
        segments = [ast.get_source_segment(text, node, padded=True) for node in chosen]
        decorated = [
            "\n".join(f"@{ast.unparse(decorator)}" for decorator in node.decorator_list)
            for node in chosen
        ]
        body = "\n\n\n".join(
            f"{decorators}\n{segment}" if decorators else segment
            for decorators, segment in zip(decorated, segments)
        )
        return f"# Extracted from {self.commit}:{self.path}\n\n\n{body}\n"


@dataclass(frozen=True)
class Fixture:
    """One historical relocation: the commit's excerpts as modules of one package."""

    name: str
    excerpts: tuple[tuple[str, Excerpt], ...]

    def write(self) -> None:
        directory = FIXTURES / self.name
        directory.mkdir(parents=True, exist_ok=True)
        for module, excerpt in self.excerpts:
            (directory / f"{module}.py").write_text(excerpt.source())


FIXTURES_82 = (
    # #104 c5b955e: the graph threaded as `SemanticDescentGraph | None` parameters.
    Fixture(
        "c5b955e_graph_parameters",
        (("cli", Excerpt("c5b955e", "nominal_refactor_advisor/cli.py", ("ColdScanAnalysis", "FastCacheScan"))),),
    ),
    # #110 (a3eed478): `is_owner` derived from the record's own fields, branched on.
    Fixture(
        "a3eed478_is_owner",
        (
            (
                "codemod_optional_ast_field",
                Excerpt(
                    "a3eed478",
                    "nominal_refactor_advisor/codemod_optional_ast_field.py",
                    ("NamedOwnerDeclaration", "ProjectDecoderOwners"),
                ),
            ),
        ),
    ),
    # #77 (ac2a002d): `keyed_metaclass_family` probing an optional field, branched on.
    Fixture(
        "ac2a002d_keyed_metaclass_family",
        (
            ("registry_identity", Excerpt("ac2a002d", "nominal_refactor_advisor/registry_identity.py", ("AutoRegisterClassAuthority",))),
            ("manual_registry", Excerpt("ac2a002d", "nominal_refactor_advisor/manual_registry.py", lines=(833,))),
        ),
    ),
    # #81 (310b9047): `spells_as_clause` is `asname is not None` renamed.
    Fixture(
        "310b9047_spells_as_clause",
        (
            ("lexical_bindings", Excerpt("310b9047", "nominal_refactor_advisor/lexical_bindings.py", ("ImportAlias",), (407,))),
            ("quoted_reference", Excerpt("310b9047", "nominal_refactor_advisor/quoted_reference.py", lines=(365, 385))),
        ),
    ),
    # #81 (3d33f21d): `declares_dataclass` answered by a constant per member.
    Fixture(
        "3d33f21d_declares_dataclass",
        (
            (
                "class_index",
                Excerpt(
                    "3d33f21d",
                    "nominal_refactor_advisor/class_index.py",
                    ("CompactDataclassFacet", "CompactDataclassDeclaration", "NoDataclassDeclaration"),
                    (2683, 2717),
                ),
            ),
        ),
    ),
    # #81 (8eef04ab): `fold_binding(bound, unbound)` called with lambdas.
    Fixture(
        "8eef04ab_fold_binding",
        (
            (
                "class_index",
                Excerpt(
                    "8eef04ab",
                    "nominal_refactor_advisor/class_index.py",
                    ("CompactNominalReference", "BoundNominalReference", "UnboundNominalReference"),
                ),
            ),
        ),
    ),
    # #81/#85 (580b14d4): the 16 `except AbsentStatementValue | UnboundReference` sites.
    Fixture(
        "580b14d4_exception_absence",
        (
            ("ast_projection", Excerpt("580b14d4", "nominal_refactor_advisor/ast_projection.py", ("AbsentStatementValue",))),
            ("class_mro", Excerpt("580b14d4", "nominal_refactor_advisor/class_mro.py", ("UnboundReference",), (231,))),
            ("class_identity_observations", Excerpt("580b14d4", "nominal_refactor_advisor/class_identity_observations.py", lines=(1542,))),
            ("class_index", Excerpt("580b14d4", "nominal_refactor_advisor/class_index.py", lines=(2072, 4817, 6980))),
            ("helpers", Excerpt("580b14d4", "nominal_refactor_advisor/detectors/_helpers.py", lines=(703, 822, 4394, 4458))),
            ("manual_registry", Excerpt("580b14d4", "nominal_refactor_advisor/manual_registry.py", lines=(149, 923))),
            ("observation_families", Excerpt("580b14d4", "nominal_refactor_advisor/observation_families.py", lines=(1110,))),
            ("product_flow", Excerpt("580b14d4", "nominal_refactor_advisor/product_flow.py", lines=(4437,))),
            ("restated_fact_observations", Excerpt("580b14d4", "nominal_refactor_advisor/restated_fact_observations.py", lines=(767,))),
            ("semantic_descent", Excerpt("580b14d4", "nominal_refactor_advisor/semantic_descent.py", lines=(5047, 5064))),
        ),
    ),
    # Main at 674a5ed9: `declared_annotations` returns `(annotation,)` or `()`.
    Fixture(
        "674a5ed9_declared_annotations",
        (("ast_field_decoders", Excerpt("674a5ed9", "nominal_refactor_advisor/ast_field_decoders.py", ("declared_annotations",))),),
    ),
)


if __name__ == "__main__":
    for fixture in FIXTURES_82:
        fixture.write()
