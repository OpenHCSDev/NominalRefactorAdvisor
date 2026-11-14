"""Scan every test fixture tree with every detector and dump each finding in full.

usage: fixture_scan.py CORPUS_DIR OUT.json [--only LIST.txt] [--workers N]
Each fixture gets a fresh cache.  The dump maps fixture name -> sorted finding JSON rows.
"""
import collections, json, shutil, sys, tempfile, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def scan(fixture: str) -> tuple[str, list[str]]:
    from nominal_refactor_advisor.analysis import analyze_path
    from nominal_refactor_advisor.json_reports import json_report_object

    root = Path(fixture)
    # the analysis cache is written beside the cache directory, so both live in one parent
    parent = Path(tempfile.mkdtemp(prefix="nra-fx-", dir=root.parent.parent))
    cache = parent / "cache"
    try:
        findings = analyze_path(root, cache_dir=cache)
        rows = sorted(
            json.dumps(dict(json_report_object(f), stable_id=f.stable_id), sort_keys=True, default=str)
            for f in findings
        )
    except BaseException as error:  # a fixture that cannot be scanned is itself a row
        rows = [f"ERROR {type(error).__name__}: {str(error)[:300]}"]
    finally:
        shutil.rmtree(parent, ignore_errors=True)
    return root.name, rows


def main() -> None:
    corpus, out = Path(sys.argv[1]), Path(sys.argv[2])
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 3
    names = sorted(p.name for p in corpus.iterdir() if p.is_dir() and not p.name.startswith("."))
    if "--only" in sys.argv:
        wanted = set(Path(sys.argv[sys.argv.index("--only") + 1]).read_text().split())
        names = [name for name in names if name in wanted]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = dict(pool.map(scan, [str(corpus / name) for name in names], chunksize=4))
    out.write_text(json.dumps(results, indent=0, sort_keys=True))
    detectors = collections.Counter()
    errors = 0
    for rows in results.values():
        for row in rows:
            if row.startswith("ERROR"):
                errors += 1
            else:
                detectors[json.loads(row).get("detector_id", "?")] += 1
    print(f"fixtures: {len(results)} | findings: {sum(detectors.values())} | detectors firing: {len(detectors)} | errors: {errors} | {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
