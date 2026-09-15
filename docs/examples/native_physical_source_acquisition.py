"""Acquire physical source candidates and authenticate their current native body.

Candidate paths come from the actual function's code and globals, not its
qualification label or import-loader callbacks. This does not prove activation.
"""

from nominal_refactor_advisor.codemod import (
    CodemodPlanSequence,
    CodemodSourceSnapshot,
    EnsureImportOperation,
    ReplaceFunctionBodyOperation,
    SourceRewriteTarget,
)

PATH = "nominal_refactor_advisor/native_compilation.py"


def physical_source_plan(snapshot: CodemodSourceSnapshot) -> CodemodPlanSequence:
    return CodemodPlanSequence.from_operations(
        (
            EnsureImportOperation(
                target=SourceRewriteTarget(file_path=PATH),
                import_source="import tokenize",
            ),
            ReplaceFunctionBodyOperation(
                target=SourceRewriteTarget(
                    file_path=PATH, qualname="NativePythonCompilation.from_function"
                ),
                body_source='''"""Authenticate physical candidates, not loader callbacks or activation."""
if type(function) is not FunctionType:
    raise ValueError("Python implementation requires an exact function")
namespace = function.__globals__
backend = NativeCreationBackend.current()
for key in namespace:
    backend.require_dictionary_key(key)
module_file = namespace["__file__"] if "__file__" in namespace else None
if module_file is not None and type(module_file) is not str:
    raise ValueError("Python source path requires an exact native string")
filename = function.__code__.co_filename
paths = dict.fromkeys((filename,) if module_file is None else (filename, module_file))
matches = []
rejection = None
for path in paths:
    try:
        with tokenize.open(path) as stream:
            source = stream.read()
    except (OSError, UnicodeError, SyntaxError):
        continue
    compilation = cls(source, filename)
    try:
        _ = compilation.function_source_span(function)
    except (ValueError, SyntaxError) as error:
        rejection = error
        continue
    if compilation not in matches:
        matches.append(compilation)
if not matches:
    if rejection is not None:
        raise rejection
    raise ValueError("Python implementation has no inspectable physical source")
if len(matches) != 1:
    raise ValueError("Python implementation has ambiguous physical source")
return matches[0]
''',
            ),
        )
    )
