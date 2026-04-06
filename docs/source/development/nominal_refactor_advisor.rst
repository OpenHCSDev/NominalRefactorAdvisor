Nominal Refactor Advisor
========================

This document records the current design and self-audit for the standalone AST-driven refactoring advisor.

Purpose
-------

The tool is not a generic clone detector. It is a structural issue classifier that tries to prescribe a
canonical refactoring pattern using an explicit nominal-architecture model.

The current implementation scans Python ASTs and emits findings for all current pattern families.

- repeated non-orthogonal method skeletons across classes
- repeated keyword-based builder calls that copy the same field mapping across sites
- repeated string-key projection dictionaries that should collapse into one authoritative schema
- repeated manual class-registration assignments that should move into a metaclass or registry base
- repeated ``hasattr`` / ``getattr(..., default)`` / ``AttributeError`` probing
- repeated string-based closed-family dispatch
- repeated inline literal dispatch that should be a registry, class family, or dataclass rule table
- manually maintained bidirectional registries
- manual fiber tags that should become host-native ``ABC`` fibers
- manually synchronized derived views that should become descriptors or properties
- class-registration flows where registration is decoupled from class creation
- duck-typed confusability where a consumer needs a nominal interface witness
- detector-local witness carriers whose shared provenance spine should live in one nominal base
- renamed witness-role slices such as ``class_name`` vs ``class_names`` that should collapse into reusable mixins


Primary Prescriptions
---------------------

The tool now has explicit coverage for Patterns 1 through 20, including:

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
- Pattern 15: staged orchestration boundary
- Pattern 16: authoritative context record
- Pattern 17: nominal strategy family
- Pattern 18: descriptor-derived view
- Pattern 19: nominal interface witness
- Pattern 20: nominal witness carrier family plus mixin enforcement for orthogonal renamed witness slices

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

Patterns 17 through 20 are the current paper-driven expansion of the original detector set:

- Pattern 17 covers closed enum/member strategy ladders where the host already offers a better nominal
  fiber decomposition through ``ABC`` hierarchies and automatic subclass registration.
- Pattern 18 covers rate-1 derived views: one authoritative source field should drive descriptor-backed or
  property-backed views rather than several stored copies that are manually resynchronized.
- Pattern 19 covers structural confusability under partial views. If a consumer only observes ``store`` and
  ``flush``, and several unrelated classes are confusable under that view, the fix is a nominal ``ABC``
  witness rather than ``Protocol``-style structural coincidence.
- Pattern 20 covers witness-carrier families inside the tool itself. If several dataclass carriers repeat the
  same provenance spine under renamed fields such as ``class_name`` vs ``function_name`` vs
  ``registry_name``, the advisor now normalizes those fields semantically, prescribes one shared nominal
  base, and can additionally force mixin extraction when orthogonal renamed slices like ``class_name`` /
  ``class_names`` need to survive together under multiple inheritance.

This semantic-role normalization matters because rate-1 architecture is not purely lexical. Several families
can carry the same witness roles under different field names. The advisor therefore distinguishes between:

- lexical field repetition (same field name, same type)
- semantic witness repetition (same provenance or focal-subject role under renamed fields)

The second case is what allows the advisor to detect that ``class_name`` and ``class_names`` belong to the
same higher-order witness family, and to prescribe a shared mixin when one carrier stores a singular name
while another stores a name family.

The advisor was first exercised on one large scientific codebase, but it is now intended as a standalone,
general-purpose tool.


Why the Design Follows the Docs
-------------------------------

The implementation itself follows the docs guidance:

- detector family uses an ``ABC`` with a concrete ``detect`` method and abstract hook for the detector
  body
- repeated per-module detector orchestration is extracted into ``PerModuleIssueDetector`` so implementation
  classes only keep orthogonal matching logic
- findings are authoritative dataclasses rather than loose dicts
- no ``Protocol`` is used as a semantic boundary
- when several classes are confusable under a consumer's partial view, the target is an ``ABC`` witness, not
  structural duck typing
- each finding reports capability gap and relation context, not just syntactic similarity

The first implementation pass exposed two problems and was immediately iterated:

- AST fingerprinting was mutating live module trees, so later detectors were reading a corrupted partial
  view; this was fixed by fingerprinting deep copies
- multiple implementation detectors repeated the same module-iteration structure; this was extracted upward
  into ``PerModuleIssueDetector`` to match the docs' ABC rule


Self-Rating
-----------

- Representation/fiber lens: good
- Capability-aware prescriptions: medium
- Provenance reporting: medium
- ABC-first prescription for repeated non-orthogonal logic: good
- Repeated field-assignment detection: good in the initial keyword-builder form
- Repeated projection-schema detection: good in the initial string-key dict form
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

The next detector-design pass should also push harder on three quality goals:

- keep detectors deterministic and explicitly certified rather than adding opaque semantic scoring
- widen lexical grouping into semantic-role normalization so coverage of true semantic duplication improves
- prefer reuse of existing nominal authorities before prescribing new synthetic bases, schemas, or mixins

The CLI now already includes richer pattern guidance for each finding:

- the pattern prescription
- the canonical target shape
- concrete first moves for the refactor
- example skeletons for the strongest canonical targets

The next iteration should move from generic example skeletons to issue-specific suggested scaffolds for the
strongest detector families, especially Patterns 5, 6, and 14.

The builder detector currently focuses on repeated keyword-constructor shapes. A later pass should widen it
to export dictionaries and repeated dataclass-to-dataclass conversion blocks.

That widening has now started: the advisor also detects repeated string-key projection dictionaries,
including export dicts and kwargs/source-value bags, and points them toward one authoritative schema.

The goal is not to produce more findings. The goal is to make each finding more canonical and less noisy.

That now includes self-hosting accuracy: when the advisor sees a class or carrier that already matches an
existing reusable nominal base, it should say so directly instead of only recommending that some new base be
introduced.
