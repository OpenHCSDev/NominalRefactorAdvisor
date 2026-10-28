"""The historical relocations of #82, read from git when a test runs.

Each relocation is a set of excerpts: whole top-level definitions of one file at
the commit that introduced a relocated probe, chosen by name or by a line they
contain, with that file's top-level imports so its names bind as they did. The
excerpts are written at their repository paths under a test directory, so a
relative import in one excerpt resolves to the class another excerpt declares.
"""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Excerpt:
    """Top-level definitions of one file at one commit, by name or by a line they contain."""

    commit: str
    path: str
    names: tuple[str, ...] = ()
    lines: tuple[int, ...] = ()

    def contains(self, node: ast.stmt) -> bool:
        return any(node.lineno <= line <= (node.end_lineno or node.lineno) for line in self.lines)

    def source(self) -> str:
        text = subprocess.run(
            ["git", "show", f"{self.commit}:{self.path}"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout
        tree = ast.parse(text)
        imports = [node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)]
        chosen = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and (node.name in self.names or self.contains(node))
        ]
        assert set(self.names) <= {node.name for node in chosen}, (self, self.names)
        definitions = [
            "\n".join(
                (*(f"@{ast.unparse(decorator)}" for decorator in node.decorator_list),
                 ast.get_source_segment(text, node, padded=True))
            )
            for node in chosen
        ]
        header = "\n".join(ast.unparse(node) for node in imports)
        return "\n\n\n".join((header, *definitions)) + "\n"


@dataclass(frozen=True)
class Relocation:
    """One historical relocation: its commit's excerpts, written at their paths."""

    excerpts: tuple[Excerpt, ...]

    def write(self, root: Path) -> Path:
        for excerpt in self.excerpts:
            target = root / excerpt.path
            for package in (target.parent, *target.parent.parents):
                if package == root:
                    break
                package.mkdir(parents=True, exist_ok=True)
                (package / "__init__.py").touch()
            target.write_text(excerpt.source())
        return root


NRA = "nominal_refactor_advisor"

# #104 c5b955e: the graph threaded as `SemanticDescentGraph | None` parameters.
GRAPH_PARAMETERS = Relocation(
    (Excerpt("c5b955e", f"{NRA}/cli.py", ("ColdScanAnalysis", "FastCacheScan")),)
)
# #110 (a3eed478): `is_owner` derived from the record's own fields, branched on.
IS_OWNER = Relocation(
    (
        Excerpt(
            "a3eed478",
            f"{NRA}/codemod_optional_ast_field.py",
            ("NamedOwnerDeclaration", "ProjectDecoderOwners"),
        ),
    )
)
# #77 (ac2a002d): `keyed_metaclass_family` probing an optional field, branched on.
KEYED_METACLASS_FAMILY = Relocation(
    (
        Excerpt("ac2a002d", f"{NRA}/registry_identity.py", ("AutoRegisterClassAuthority",)),
        Excerpt("ac2a002d", f"{NRA}/manual_registry.py", lines=(833,)),
    )
)
# #81 (310b9047): `spells_as_clause` is `asname is not None` renamed.
SPELLS_AS_CLAUSE = Relocation(
    (
        Excerpt("310b9047", f"{NRA}/lexical_bindings.py", ("ImportAlias",), (407,)),
        Excerpt("310b9047", f"{NRA}/quoted_reference.py", lines=(365, 385)),
    )
)
# #81 (3d33f21d): `declares_dataclass` answered by a constant per member.
DECLARES_DATACLASS = Relocation(
    (
        Excerpt(
            "3d33f21d",
            f"{NRA}/class_index.py",
            ("CompactDataclassFacet", "CompactDataclassDeclaration", "NoDataclassDeclaration"),
            (2683, 2717),
        ),
    )
)
# #81 (8eef04ab): `fold_binding(bound, unbound)` called with lambdas.
FOLD_BINDING = Relocation(
    (
        Excerpt(
            "8eef04ab",
            f"{NRA}/class_index.py",
            ("CompactNominalReference", "BoundNominalReference", "UnboundNominalReference"),
        ),
    )
)
# #81/#85 (580b14d4): the 16 `except AbsentStatementValue | UnboundReference` sites.
EXCEPTION_ABSENCE = Relocation(
    (
        Excerpt("580b14d4", f"{NRA}/ast_projection.py", ("AbsentStatementValue",)),
        Excerpt("580b14d4", f"{NRA}/class_mro.py", ("UnboundReference",), (231,)),
        Excerpt("580b14d4", f"{NRA}/class_identity_observations.py", lines=(1542,)),
        Excerpt("580b14d4", f"{NRA}/class_index.py", lines=(2072, 4817, 6980)),
        Excerpt("580b14d4", f"{NRA}/detectors/_helpers.py", lines=(703, 822, 4394, 4458)),
        Excerpt("580b14d4", f"{NRA}/manual_registry.py", lines=(149, 923)),
        Excerpt("580b14d4", f"{NRA}/observation_families.py", lines=(1110,)),
        Excerpt("580b14d4", f"{NRA}/product_flow.py", lines=(4437,)),
        Excerpt("580b14d4", f"{NRA}/restated_fact_observations.py", lines=(767,)),
        Excerpt("580b14d4", f"{NRA}/semantic_descent.py", lines=(5047, 5064)),
    )
)
# Main at 674a5ed9: `declared_annotations` returns `(annotation,)` or `()`.
DECLARED_ANNOTATIONS = Relocation(
    (Excerpt("674a5ed9", f"{NRA}/ast_field_decoders.py", ("declared_annotations",)),)
)
