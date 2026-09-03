# Nominal Refactor Advisor

AST-driven refactoring advisor for nominal architecture, SSOT recovery, and
anti-duck-typing cleanup in mature Python systems.

NRA is designed for repositories whose production behaviour, integrations,
and operational knowledge make replacement prohibitively expensive.  A
practitioner or agent chooses the intended semantic authority.  NRA proves the
current source boundary and compiles the deterministic dependency, import,
placement, and consumer-rewrite work into one reviewable change.

The tool emits evidence-backed refactoring findings, architectural direction,
and proof-gated executable recipes where the source establishes a safe target.

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

- pattern docs are generated from `PatternId` declarations
- detector docs are generated from the registered `IssueDetector` family

Start with:

- `docs/source/api/getting_started.rst`
- `docs/source/api/pattern_catalog.rst`
- `docs/source/api/detector_catalog.rst`
