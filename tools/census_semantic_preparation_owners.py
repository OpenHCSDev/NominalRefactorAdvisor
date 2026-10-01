"""Original syntax/class-family census; unmatched declarations remain OPEN."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--advisor-root", required=True, type=Path)
    parser.add_argument("sources", nargs="+", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.advisor_root))
    from nominal_refactor_advisor.ast_tools import (
        PythonModuleRootParser,
        module_syntax_index,
    )
    from nominal_refactor_advisor.class_index import (
        CompactModuleClassProjectionFamily,
        build_compact_class_family_index,
    )
    from nominal_refactor_advisor.analysis import release_module_analysis_memory

    for path in args.sources:
        module = PythonModuleRootParser.for_root(
            path, use_parse_cache=False, parse_workers=1
        ).parsed_source_path(path)
        syntax = module_syntax_index(module.module)
        indexed = tuple(
            build_compact_class_family_index(
                CompactModuleClassProjectionFamily.collect(module)
            ).classes_by_symbol.values()
        )
        for index, node in syntax.indexed_nodes_of_type(ast.ClassDef):
            matches = tuple(owner for owner in indexed if owner.line == node.lineno)
            print(
                json.dumps(
                    {
                        "file_path": str(path),
                        "line": node.lineno,
                        "name": node.name,
                        "scope": syntax.scopes[syntax.scope_ids[index]].names,
                        "admitted_owners": [owner.symbol for owner in matches],
                        "status": (
                            "INDEX_ADMITTED"
                            if len(matches) == 1
                            else "OPEN_UNPROJECTED_OR_AMBIGUOUS"
                        ),
                    }
                ),
                flush=True,
            )
        del module, syntax, indexed
        release_module_analysis_memory()


if __name__ == "__main__":
    main()
