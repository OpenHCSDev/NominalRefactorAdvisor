"""Renderer helpers copied verbatim from codemod.py at b849d95.

Only the two methods are retained; imports and the executable probe are fixture
scaffolding. The full historical module is replayed separately from Git.
"""

from __future__ import annotations

import ast

from nominal_refactor_advisor.detectors._base import (
    CallableCandidateFindingRenderer,
    DetectorDeclaration,
    DirectBuildFindingRendererCandidate,
)


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


if __name__ == "__main__":
    candidate = DirectBuildFindingRendererCandidate(
        file_path=__file__, line=1, class_name="ProbeDetector",
        method_name="_finding_for_candidate",
        base_name="ModuleCollectorCandidateDetector",
        parameter_name="candidate", positional_arg_count=2, keyword_names=(),
    )
    call = ast.parse(
        "self.build_finding(candidate.parameter_name, ())", mode="eval",
    ).body
    rendered = DirectBuildFindingRendererFindingRecipeSynthesizer.renderer_source(
        candidate, call,
    )
    print(rendered)

