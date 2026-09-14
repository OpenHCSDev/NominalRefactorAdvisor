"""Changed interpreter hooks cannot nominate themselves as trusted protocols."""

import subprocess
import sys
from textwrap import dedent

import pytest


@pytest.mark.parametrize("hook", ("__import__", "__build_class__"))
def test_changed_hook_before_entry_is_not_its_own_protocol_authority(hook):
    script = dedent("""
        import ast
        import builtins
        from pathlib import Path
        from nominal_refactor_advisor.ast_tools import ParsedModule
        from nominal_refactor_advisor.source_execution import SourceModuleExecution

        hook = HOOK
        original = getattr(builtins, hook)
        def replacement(*args, **kwargs):
            return original(*args, **kwargs)
        replacement.__name__ = original.__name__
        source = "import builtins\\n" if hook == "__import__" else "class Owner: pass\\n"
        module = ParsedModule(Path("hook.py"), "hook", False, ast.parse(source), source)
        def require_protocol():
            execution = SourceModuleExecution.from_module(module)
            if hook == "__import__":
                execution.require_import(module.module.body[0])
            else:
                execution.require_class_creation(module.module.body[0])

        require_protocol()
        setattr(builtins, hook, replacement)
        try:
            try:
                require_protocol()
            except ValueError as error:
                assert str(error) == "Captured object is not the required native declaration", error
            else:
                raise AssertionError("Changed hook was admitted as its own native authority")
        finally:
            setattr(builtins, hook, original)
        require_protocol()
    """).replace("HOOK", repr(hook))
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
