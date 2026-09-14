"""Implicit documentation must retain the compiler's actual text production."""

import json
import subprocess
import sys

import pytest

from test_documentation_store import execution


@pytest.mark.parametrize(
    "source",
    (
        '"""Summary.\n        Details.\n    """\nsaved = __doc__\n',
        'class Owner:\n    """Summary.\n        Details.\n    """\n    saved = __doc__\n',
        'class Owner:\n    """Summary.\n        Details.\n    """\n    global __doc__\n    saved = __doc__\n',
        'class Owner:\n    __doc__ = """Summary.\n        Details.\n    """\n    saved = __doc__\n',
    ),
)
def test_captured_documentation_equals_native_final_text(source):
    class_scope = source.startswith("class")
    expression = "Owner.saved" if class_scope else "saved"
    native = subprocess.run(
        [
            sys.executable,
            "-c",
            source + f"\nimport json; print(json.dumps({expression}))\n",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    env = execution(source)
    if class_scope:
        owner = env.module.module.body[0]
        env.require_class_creation(owner)
        captured = env.class_entry(owner).completion_member("saved")
    else:
        captured = env.capture_value(env.module.module.body[-1].value)
    assert captured.require_native_text() == json.loads(native.stdout)
