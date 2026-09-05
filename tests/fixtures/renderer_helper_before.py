"""Renderer helpers copied verbatim from codemod.py at b849d95.

Only the two methods are retained; imports and the executable probe are fixture
scaffolding. The full historical module is replayed separately from Git.
"""

from __future__ import annotations

import ast
from pathlib import Path

from nominal_refactor_advisor.ast_tools import parse_python_modules
from nominal_refactor_advisor.codemod import (
    CodemodSourceSnapshot,
    DeclareCandidateFindingRendererOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.detectors._base import (
    CallableCandidateFindingRenderer,
    DetectorDeclaration,
    DirectBuildFindingRendererCandidate,
    ModuleCollectorCandidateDetector,
)


class ProbeCandidate:
    pass


class ProbeDetector(ModuleCollectorCandidateDetector[ProbeCandidate]):
    detector_id = "renderer_replay_probe"
    candidate_collector = staticmethod(lambda module: ())

    def _finding_for_candidate(self, candidate):
        return self.build_finding(candidate.parameter_name, ())


class DirectBuildFindingRendererFindingRecipeSynthesizer:
    @staticmethod
    def renderer_lambda(parameter_name: str, value: ast.expr) -> ast.Lambda:
        return ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self"), ast.arg(arg=parameter_name)],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=value,
        )

    @classmethod
    def renderer_source(
        cls,
        candidate: DirectBuildFindingRendererCandidate,
        call: ast.Call,
    ) -> str:
        assignment = ast.Assign(
            targets=[
                ast.Name(
                    id=DetectorDeclaration.finding_renderer_field_name,
                    ctx=ast.Store(),
                )
            ],
            value=ast.Call(
                func=ast.Name(
                    id=CallableCandidateFindingRenderer.__name__,
                    ctx=ast.Load(),
                ),
                args=[cls.renderer_lambda(candidate.parameter_name, call)],
                keywords=[],
            ),
        )
        return ast.unparse(ast.fix_missing_locations(assignment))


def run():
    path = Path(__file__)
    snapshot = CodemodSourceSnapshot.from_modules(parse_python_modules(path))
    witness = DeclareCandidateFindingRendererOperation(
        target=SourceRewriteTarget(
            file_path=str(path), qualname="ProbeDetector._finding_for_candidate",
        ),
    ).required_witness(snapshot)
    candidate = witness.candidate
    call = candidate.build_call(witness.node)
    rendered = DirectBuildFindingRendererFindingRecipeSynthesizer.renderer_source(
        candidate, call,
    )
    print(rendered)


if __name__ == "__main__":
    run()
