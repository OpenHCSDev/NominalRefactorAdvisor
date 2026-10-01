"""Measure the original family on fixed source files; not a detector/audit copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--advisor-root", required=True, type=Path)
    parser.add_argument("sources", nargs="+", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.advisor_root))
    import nominal_refactor_advisor
    from nominal_refactor_advisor.analysis import release_module_analysis_memory
    from nominal_refactor_advisor.ast_tools import (
        PythonModuleRootParser,
        collected_family_items_content_signature,
        retains_python_ast,
    )
    from nominal_refactor_advisor.semantic_descent import (
        CompactSemanticModuleProjectionFamily,
    )

    assert Path(nominal_refactor_advisor.__file__).is_relative_to(args.advisor_root)
    for path in args.sources:
        module = PythonModuleRootParser.for_root(
            path, use_parse_cache=False, parse_workers=1
        ).parsed_source_path(path)
        started = perf_counter()
        items = tuple(CompactSemanticModuleProjectionFamily.collect(module))
        elapsed = perf_counter() - started
        assert not retains_python_ast(items)
        print(
            json.dumps(
                {
                    "path": str(path),
                    "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "content_signature": collected_family_items_content_signature(
                        items
                    ),
                    "collection_seconds": elapsed,
                    "presentations": sum(len(item.projections) for item in items),
                    "checks": sum(len(item.type_checks.checks) for item in items),
                    "supplements": sum(len(item.class_supplements) for item in items),
                }
            ),
            flush=True,
        )
        del module, items
        release_module_analysis_memory()


if __name__ == "__main__":
    main()
