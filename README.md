# Nominal Refactor Advisor

AST-driven refactoring advisor for nominal architecture and anti-duck-typing cleanup.

This tool detects structural issues such as repeated method families, manual class registration,
attribute-probe dispatch, repeated projection dicts, repeated builder mappings, and bidirectional
registries, then prescribes canonical refactor patterns grounded in the nominal-architecture docs.

The current implementation grew out of DQ-Dock, but the anti-patterns it targets are universal.

Run locally with:

```bash
python -m dq_dock_engine.refactor_advisor path/to/python/package
```
