# Nominal Refactor Advisor

AST-driven refactoring advisor for nominal architecture and anti-duck-typing cleanup.

It detects structural issues such as:

- repeated method families that want an `ABC`
- manual class registration that wants a metaclass or registry base
- repeated builder/projection mappings that violate SSOT
- attribute-probe dispatch that should become nominal dispatch
- bidirectional registry patterns that want one authoritative bijection

For each finding it emits:

- the prescribed canonical pattern
- the target architectural shape
- first refactor moves
- example skeletons
- issue-specific scaffolds and codemod-style patch suggestions for the strongest patterns

Run locally with:

```bash
python -m nominal_refactor_advisor path/to/python/package
```
