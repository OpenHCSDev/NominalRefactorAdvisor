# Nominal Refactor Advisor

AST-driven refactoring advisor for nominal architecture, SSOT recovery, and
anti-duck-typing cleanup.

The tool emits refactoring findings backed by canonical patterns, architectural
prescriptions, first moves, and stronger scaffolds where the pattern supports
them.

Run locally with:

```bash
nominal-refactor-advisor path/to/python/package
```

Build the Sphinx docs with:

```bash
pip install -e .[docs]
python -m sphinx -b html docs/source docs/_build/html
```

The docs are intentionally code-derived where possible:

- pattern docs are generated from `PATTERN_SPECS`
- detector docs are generated from the registered `IssueDetector` family

Start with:

- `docs/source/api/getting_started.rst`
- `docs/source/api/pattern_catalog.rst`
- `docs/source/api/detector_catalog.rst`
