Nominal Refactor Advisor
========================

This document records the current design and self-audit for the standalone AST-driven refactoring advisor.

Purpose
-------

The tool is not a generic clone detector. It is a structural issue classifier that tries to prescribe a
canonical refactoring pattern using the nominal-architecture rules in this docs set.

The current implementation scans Python ASTs and emits findings for all current pattern families.

- repeated non-orthogonal method skeletons across classes
- repeated keyword-based builder calls that copy the same field mapping across sites
- repeated string-key projection dictionaries that should collapse into one authoritative schema
- repeated manual class-registration assignments that should move into a metaclass or registry base
- repeated ``hasattr`` / ``getattr(..., default)`` / ``AttributeError`` probing
- repeated string-based closed-family dispatch
- repeated inline literal dispatch that should be a registry, class family, or dataclass rule table
- manually maintained bidirectional registries


Primary Prescriptions
---------------------

The tool now has explicit coverage for Patterns 1 through 14, including:

- Pattern 1: sentinel attribute simulation vs nominal boundary
- Pattern 2: discriminated-union predicate chains
- Pattern 3: closed-family O(1) dispatch
- Pattern 4: polymorphic configuration contracts
- Pattern 5: ABC template-method migration
- Pattern 6: AutoRegisterMeta / class-level registration normal form
- Pattern 7: generated-type lineage tracking
- Pattern 8: dual-axis scope x type resolution
- Pattern 9: custom ``isinstance`` / virtual membership
- Pattern 10: dynamic interface generation
- Pattern 11: sentinel type capability markers
- Pattern 12: dynamic method injection into type namespaces
- Pattern 13: bidirectional type lookup
- Pattern 14: authoritative constructor / projection schema

Pattern 5 should be read broadly: the advisor may point toward an ``ABC`` plus concrete implementation
classes, or toward an ``ABC`` plus mixins when some concerns are orthogonal but still belong in the
nominal MRO-governed hierarchy rather than in composition wrappers.

Pattern 6 (not yet implemented as a detector) should be used when the problem is fundamentally class-level:

Pattern 6 is now implemented for three common manual-registration shapes:

- repeated ``REGISTRY[key] = Class`` assignments
- repeated helper calls like ``registry.register(Class, key)``
- repeated decorator registration like ``@register(REGISTRY, key)``

- import-time registration
- abstract-class skipping
- registry inheritance across class families
- uniqueness checks on declared class identity
- discovery based on the class object itself rather than instances

That is the ``AutoRegisterMeta`` case. It is the class-level canonical analogue of Pattern 5: shared
non-orthogonal registration logic lives in one authoritative metaclass, while orthogonal differences remain
as declarative class hooks.

For Pattern 3, the prescription is intentionally broad: the replacement can be an enum-keyed registry, a
class-registration family, or a dataclass-backed rule table. The important point is that the cases become
data in one authoritative structure rather than repeated literal branches in code.

DQ-Dock was the initial proving ground, but the advisor is no longer DQ-Dock-specific.


Why the Design Follows the Docs
-------------------------------

The implementation itself follows the docs guidance:

- detector family uses an ``ABC`` with a concrete ``detect`` method and abstract hook for the detector
  body
- repeated per-module detector orchestration is extracted into ``PerModuleIssueDetector`` so implementation
  classes only keep orthogonal matching logic
- findings are authoritative dataclasses rather than loose dicts
- no ``Protocol`` is used as a semantic boundary
- each finding reports capability gap and relation context, not just syntactic similarity

The first implementation pass exposed two problems and was immediately iterated:

- AST fingerprinting was mutating live module trees, so later detectors were reading a corrupted partial
  view; this was fixed by fingerprinting deep copies
- multiple implementation detectors repeated the same module-iteration structure; this was extracted upward
  into ``PerModuleIssueDetector`` to match the docs' ABC rule


Self-Rating Against the Docs
----------------------------

Against ``nominal_architecture_playbook.rst``:

- Representation/fiber lens: good
- Capability-aware prescriptions: medium
- Provenance reporting: medium
- Unit-rate / authoritative-owner analysis: weak in MVP, needs a dedicated detector

Against ``agent_refactoring_crash_course.rst``:

- Mechanical audit support: good
- ABC-first prescription for repeated non-orthogonal logic: good
- Repeated field-assignment detection: good in the initial keyword-builder form
- Repeated projection-schema detection: good in the initial string-key dict form
- Orthogonal-core reduction beyond direct clones: medium

Against ``nominal_identity_case_studies.rst``:

- Pattern coverage: complete at the first heuristic level
- Confidence quality: mixed; some patterns are stronger than others and still need sharpening


Next Iteration Targets
----------------------

The next useful iterations are no longer about missing pattern IDs. They are about improving precision,
reducing false positives, and adding codemod suggestions.

1. sharpen Pattern 3 so constant maps are separated from true dispatch smells
2. widen Pattern 14 to dataclass-to-dataclass conversion blocks
3. add suggested refactor skeletons and codemod scaffolds, not just findings
4. add repository configuration and suppression support
5. extract more generic docs into the standalone repo over time

The CLI now already includes richer pattern guidance for each finding:

- the pattern prescription
- the canonical target shape
- concrete first moves for the refactor
- example skeletons for the strongest canonical targets

The builder detector currently focuses on repeated keyword-constructor shapes. A later pass should widen it
to export dictionaries and repeated dataclass-to-dataclass conversion blocks.

That widening has now started: the advisor also detects repeated string-key projection dictionaries,
including export dicts and kwargs/source-value bags, and points them toward one authoritative schema.

The goal is not to produce more findings. The goal is to make each finding more canonical and less noisy.
