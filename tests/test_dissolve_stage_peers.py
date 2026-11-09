"""Shells that dissolve in one stage leave each other alone: each deletes itself and
sends its own consumers to the modules that declare what they import."""

from __future__ import annotations

from pathlib import Path

from tests.test_package_move_batch import _moved, _runs, _write


def test_two_shells_re_exporting_each_other_dissolve_in_one_stage(tmp_path: Path) -> None:
    """`app` imports `X` from `a`, which re-exports it from `b`, which re-exports it
    from `real`. Both shells dissolve in the same stage. `b`'s dissolve used to
    rewrite `a`, the file `a`'s own dissolve deletes: the stage conflicted and the
    whole table moved nothing. Now `app` reads `X` from its declaring module."""

    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/messaging/__init__.py": "",
            "pkg/real.py": "class X:\n    pass\n",
            "pkg/b.py": "from pkg.real import X\n",
            "pkg/a.py": "from pkg.b import X\n",
            "pkg/app.py": "from pkg.a import X\n\nVALUE = X.__name__\n",
        },
    )

    _stages, root = _moved(
        tmp_path,
        "| `messaging` | `real.py` | |\n| `messaging` | `b.py` | |\n| `messaging` | `a.py` | |\n",
    )
    assert not (root / "pkg/a.py").exists()
    assert not (root / "pkg/b.py").exists()
    assert (root / "pkg/app.py").read_text() == (
        "from pkg.messaging.real import X\n\nVALUE = X.__name__\n"
    )
    assert _runs(root, "VALUE") == "X"
