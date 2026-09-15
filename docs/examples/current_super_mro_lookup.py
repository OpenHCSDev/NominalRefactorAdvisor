"""Derive current parent-member selection through the existing native MRO owner.

The authored plan adds an operand-selection contract, not super activation,
descriptor execution or generated-class construction evidence.
"""

import ast

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    PatchTargetOperation,
    ReplaceFunctionBodyOperation,
    ReplaceFunctionSignatureOperation,
    SourceRewriteTarget,
)
from nominal_refactor_advisor.codemod_source_edits import (
    SourceTextGeometry,
    SourceTextSpan,
)

PATH = "nominal_refactor_advisor/native_class_mro.py"


def current_super_mro_plan(snapshot: CodemodSourceSnapshot) -> CodemodPlanSequence:
    lookup = SourceRewriteTarget(
        file_path=PATH, qualname="NativeClassMroDeclaration.member_owner"
    )
    constructor = SourceRewriteTarget(
        file_path=PATH, qualname="NativeClassMroDeclaration.python_constructor"
    )
    identifier = constructor.required_target_id(snapshot.source_index)
    (selection,) = (
        node
        for node in ast.walk(snapshot.ast_target_nodes_by_id[identifier])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "member_owner"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    )
    source = snapshot.parsed_module_for_source_path(PATH).source
    span = SourceTextSpan.from_offsets(
        SourceTextGeometry(source).required_node_offsets(selection)
    )
    return CodemodPlanSequence.from_operations(
        (
            ReplaceFunctionSignatureOperation(
                target=lookup,
                signature_suffix="(self, name: str, *, start_after: type | None = None) -> type | None:",
            ),
            ReplaceFunctionBodyOperation(
                target=lookup,
                body_source='''"""Select current stored C3 members; descriptor execution remains unproved."""
if type(name) is not str:
    raise ValueError("Lookup requires an exact native member name")
mro = self.native_mro(self.declaration)
if start_after is not None:
    starts = tuple(index for index, owner in enumerate(mro) if owner is start_after)
    if len(starts) != 1:
        raise ValueError("Native MRO lookup start owner is absent or ambiguous")
    mro = mro[starts[0] + 1:]
return next(
    (owner for owner in mro if name in self.stored_namespace(owner)),
    None,
)
''',
            ),
            ReplaceFunctionSignatureOperation(
                target=constructor,
                signature_suffix="(self, *, start_after: type | None = None) -> FunctionType:",
            ),
            PatchTargetOperation(
                target=constructor,
                replacements=(
                    span.replacement(
                        source,
                        'self.member_owner("__new__", start_after=start_after)',
                    ),
                ),
            ),
        )
    )
