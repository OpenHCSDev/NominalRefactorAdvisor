from __future__ import annotations

import pytest

from nominal_refactor_advisor.codemod import (
    CodemodPlanDocument,
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    DeleteTargetOperation,
    ForbidInheritanceOperation,
    RefactorRecipe,
    ReplaceDeclaredCallTargetOperation,
    RequireAbsentDeclarationOperation,
    RequireCallOutsideIterationOperation,
    RequireDeclaredCallOperation,
    RequireInheritanceOperation,
    SelectionCountExpectation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_preflight import CodemodOperationPreflightError

PATH = "/repo/subject.py"
SOURCE = '''\
class AlphaAuthority:
    @staticmethod
    def run(value):
        return value


class Legacy:
    @staticmethod
    def run(value):
        return value


class Consumer:
    def render(self, value):
        return Legacy.run(value)
'''


def target(qualname: str | None = None, *, path: str = PATH) -> SourceRewriteTarget:
    return SourceRewriteTarget(file_path=path, qualname=qualname)


def verification_document() -> CodemodPlanDocument:
    return CodemodPlanDocument(
        recipes=(
            RefactorRecipe(
                "verify-render-owner",
                operations=(
                    RequireDeclaredCallOperation(
                        target=target("Consumer.render"),
                        callee=target("AlphaAuthority.run"),
                        selection_count=SelectionCountExpectation(exact=1),
                    ),
                    RequireAbsentDeclarationOperation(
                        target=target(),
                        forbidden=target("Legacy"),
                    ),
                ),
            ),
        )
    )


def test_existing_preflight_reports_every_failed_positive_requirement():
    snapshot = CodemodSourceSnapshot.from_source_mapping({PATH: SOURCE})

    with pytest.raises(CodemodOperationPreflightError) as error:
        verification_document().simulate(snapshot)

    reports = error.value.reports
    assert [report.operation for report in reports] == [
        "require_declared_call",
        "require_absent_declaration",
    ]
    assert all(report.status.is_failed for report in reports)
    assert "No resolved calls" in reports[0].message
    assert "Competing declaration" in reports[1].message


def test_final_verification_stage_proves_route_and_retired_competitor():
    snapshot = CodemodSourceSnapshot.from_source_mapping({PATH: SOURCE})
    migration = CodemodPlanSequence.from_operations(
        (
            ReplaceDeclaredCallTargetOperation(
                target=target("Consumer.render"),
                callee=target("Legacy.run"),
                expression_source="AlphaAuthority.run",
                selection_count=SelectionCountExpectation(exact=1),
            ),
            DeleteTargetOperation(target=target("Legacy")),
        )
    )
    sequence = CodemodPlanSequence.compose((migration, verification_document()))

    result = sequence.simulate(snapshot)

    assert result.is_clean
    final = result.stage_reports[-1].document_simulation
    reports = final.preflight_report.reports
    assert [report.operation for report in reports] == [
        "require_declared_call",
        "require_absent_declaration",
    ]
    assert all(report.status.is_passed for report in reports)
    assert final.simulation.applied_rewrite_count == 0


def test_inheritance_requirements_use_the_resolved_native_class_graph():
    path = "/repo/inheritance.py"
    snapshot = CodemodSourceSnapshot.from_source_mapping(
        {
            path: (
                "class FileRole: pass\n"
                "class IterationRole: pass\n"
                "class Reader(FileRole, IterationRole): pass\n"
                "class Writer(FileRole): pass\n"
            )
        }
    )
    passed = CodemodPlanDocument(
        recipes=(
            RefactorRecipe(
                "verify-roles",
                operations=(
                    RequireInheritanceOperation(
                        target=target("Reader", path=path),
                        provider=target("IterationRole", path=path),
                    ),
                    ForbidInheritanceOperation(
                        target=target("Writer", path=path),
                        provider=target("IterationRole", path=path),
                    ),
                ),
            ),
        )
    ).simulate(snapshot)

    assert passed.is_clean
    assert all(report.status.is_passed for report in passed.preflight_report.reports)

    failed = CodemodPlanDocument(
        recipes=(
            RefactorRecipe(
                "reject-writer-iteration",
                operations=(
                    ForbidInheritanceOperation(
                        target=target("Reader", path=path),
                        provider=target("IterationRole", path=path),
                    ),
                ),
            ),
        )
    ).preflight_snapshot(snapshot)
    assert failed.preflight_failed
    assert "still inherits" in failed.failed_reports[0].message


def test_iteration_requirement_distinguishes_once_repeated_and_iterable_calls():
    path = "/repo/iteration.py"
    snapshot = CodemodSourceSnapshot.from_source_mapping(
        {
            path: (
                "def expensive(value=()): return value\n\n"
                "def once(value): return expensive(value)\n\n"
                "def repeated(values):\n"
                "    return [expensive(value) for value in values]\n\n"
                "def iterable_once():\n"
                "    for value in expensive():\n"
                "        pass\n"
            )
        }
    )
    provider = target("expensive", path=path)
    clean = CodemodPlanDocument(
        recipes=(
            RefactorRecipe(
                "verify-once",
                operations=(
                    RequireCallOutsideIterationOperation(
                        target=target("once", path=path),
                        callee=provider,
                        selection_count=SelectionCountExpectation(exact=1),
                    ),
                    RequireCallOutsideIterationOperation(
                        target=target("iterable_once", path=path),
                        callee=provider,
                        selection_count=SelectionCountExpectation(exact=1),
                    ),
                ),
            ),
        )
    ).simulate(snapshot)
    assert clean.is_clean

    failed = CodemodPlanDocument(
        recipes=(
            RefactorRecipe(
                "reject-repeated",
                operations=(
                    RequireCallOutsideIterationOperation(
                        target=target("repeated", path=path),
                        callee=provider,
                        selection_count=SelectionCountExpectation(exact=1),
                    ),
                ),
            ),
        )
    ).preflight_snapshot(snapshot)
    assert failed.preflight_failed
    assert "repeated regions" in failed.failed_reports[0].message
