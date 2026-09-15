"""Project namespace-key annotations to the new key contract through the DSL.

The selected modules' NativeScalar annotations denote dictionary slots except
the explicitly retained scalar-value query. Constants and scalar producers in
the compiler retain their narrower contract. This is an authored syntax plan,
not a claim that changing an annotation proves dictionary operation behavior.
"""

import ast
from itertools import accumulate

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    PatchTargetOperation,
    SourceRewriteTarget,
    SourceTextReplacement,
)
from nominal_refactor_advisor.source_geometry import SourceByteSpan

KEY_MODULES = (
    "nominal_refactor_advisor/captured_reference.py",
    "nominal_refactor_advisor/source_entry.py",
    "nominal_refactor_advisor/source_execution.py",
    "nominal_refactor_advisor/native_call.py",
)


def key_annotation_plan(snapshot: CodemodSourceSnapshot) -> CodemodPlanSequence:
    operations = []
    for path in KEY_MODULES:
        module = snapshot.parsed_module_for_source_path(path)
        source = module.source
        retained = {
            node.returns
            for node in ast.walk(module.module)
            if isinstance(node, ast.FunctionDef)
            and node.name == "require_native_scalar"
        }
        names = tuple(
            node
            for node in ast.walk(module.module)
            if isinstance(node, ast.Name)
            and node.id == "NativeScalar"
            and node not in retained
        )
        lines = tuple(source.splitlines(keepends=True))
        offsets = tuple(accumulate((len(line) for line in lines), initial=0))
        ranges = sorted(
            SourceByteSpan.require_node(node).character_offsets(lines, offsets)
            for node in names
        )
        rewritten = source
        for start, end in reversed(ranges):
            assert source[start:end] == "NativeScalar"
            rewritten = rewritten[:start] + "NativeDictionaryKey" + rewritten[end:]
        if not retained:
            rewritten = SourceTextReplacement(
                "    NativeScalar,\n", "    NativeDictionaryKey,\n"
            ).apply_exactly_once(rewritten, subject=path)
        if path.endswith("native_call.py"):
            rewritten = SourceTextReplacement(
                "    NativeScalarValueABC,\n", ""
            ).apply_exactly_once(rewritten, subject=path)
        operations.append(
            PatchTargetOperation(
                target=SourceRewriteTarget(file_path=path),
                replacements=(SourceTextReplacement(source, rewritten),),
            )
        )
    return CodemodPlanSequence.from_operations(operations)
