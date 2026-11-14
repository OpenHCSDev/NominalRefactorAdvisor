"""pytest plugin: keep a copy of every Python source tree a test builds under tmp_path.

Usage: NRA_FIXTURE_OUT=DIR PYTHONPATH=tools/proof pytest -p snap_fixtures tests/...

The copies are a corpus of small programs that fire the detectors the main corpus never
fires; tools/proof/fixture_scan.py scans them.
"""
import hashlib
import os
import shutil
from pathlib import Path


import pytest

OUT = Path(os.environ["NRA_FIXTURE_OUT"])

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item):
    tmp = getattr(item, "funcargs", {}).get("tmp_path")
    if tmp is None or not Path(tmp).exists():
        return
    tmp = Path(tmp)
    files = sorted(
        p for p in tmp.rglob("*.py")
        if p.is_file() and not any(part.startswith(".") or part == "__pycache__" for part in p.relative_to(tmp).parts)
    )
    if not files or len(files) > 60:
        return
    digest = hashlib.sha256()
    for p in files:
        digest.update(str(p.relative_to(tmp)).encode() + b"\0" + p.read_bytes() + b"\0")
    dest = OUT / digest.hexdigest()[:16]
    if dest.exists():
        return
    staging = OUT / f".{dest.name}.{os.getpid()}"
    for p in files:
        target = staging / p.relative_to(tmp)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    (staging / "_test_id.txt").write_text(item.nodeid)
    try:
        staging.rename(dest)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
