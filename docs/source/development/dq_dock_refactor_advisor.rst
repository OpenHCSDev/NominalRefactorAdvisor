DQ-Dock Refactor Advisor
========================

This document records the initial plan and self-audit for the AST-driven refactoring advisor introduced
for DQ-Dock.

Purpose
-------

The tool is not a generic clone detector. It is a structural issue classifier that tries to prescribe a
canonical refactoring pattern using the nominal-architecture rules in this docs set.

The current MVP scans Python ASTs and emits findings for several issue families:

- repeated non-orthogonal method skeletons across classes
- repeated keyword-based builder calls that copy the same field mapping across sites
- repeated export dictionaries that should collapse into one declarative export schema
- repeated manual class-registration assignments that should move into a metaclass or registry base
- repeated ``hasattr`` / ``getattr(..., default)`` / ``AttributeError`` probing
- repeated string-based closed-family dispatch
- repeated inline literal dispatch that should be a registry, class family, or dataclass rule table
- manually maintained bidirectional registries


Primary Prescriptions
---------------------

The tool currently prescribes these case-study patterns:

- Pattern 5: ABC template-method migration
- Pattern 6: AutoRegisterMeta / class-level registration normal form
- Pattern 3: closed-family O(1) dispatch
- Pattern 13: bidirectional type lookup
- Pattern 14: authoritative constructor / shared builder

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

This matches the current DQ-Dock need: find duplicated behavior-family logic, weak structural dispatch,
and mirrored registry state before those patterns spread further.


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
- Repeated export-schema detection: good in the initial string-key export-dict form
- Orthogonal-core reduction beyond direct clones: medium

Against ``nominal_identity_case_studies.rst``:

- Pattern 3 coverage: good
- Pattern 5 coverage: good
- Pattern 13 coverage: good
- Pattern 6/7/8/9/10/11/12 coverage: not yet implemented


Next Iteration Targets
----------------------

The next useful detectors should be:

1. repeated field-assignment / record-builder detection
2. repeated subclass registration boilerplate -> Pattern 6
3. sentinel-attribute simulation of identity -> Pattern 1 or 11
4. repeated lineage-normalization logic -> Pattern 7 or 13
5. scope x type precedence detection -> Pattern 8

The builder detector currently focuses on repeated keyword-constructor shapes. A later pass should widen it
to export dictionaries and repeated dataclass-to-dataclass conversion blocks.

That widening has now started: the advisor also detects repeated string-key export dictionaries and points
them toward one authoritative export schema.

The goal is not to produce more findings. The goal is to make each finding more canonical and less noisy.
