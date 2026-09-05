Detection Substrate
===================

This page documents the internal detector framework used to extend the advisor.
It is reference material for maintainers, not the primary end-user API.

Detector Base Classes
---------------------

.. automodule:: nominal_refactor_advisor.detectors
   :members: DetectorConfig, IssueDetector, PerModuleIssueDetector, CandidateFindingDetector, EvidenceOnlyPerModuleDetector, StaticModulePatternDetector, default_detectors

Detector declarations that can assess a finding inherit
``FindingRecipeEvaluator``. Only declarations capable of producing a rewrite
also inherit ``FindingRecipeSynthesizer`` and one nominal ``RefactorConcept``.
Recipe lookup resolves the finding's registered ``IssueDetector`` declaration
through its MRO; there is no second detector-to-evaluator registry.
An unregistered finding has no proved required-relation owner and therefore no
recipe evaluator. Semantic-mirror detector declarations inherit a shared
evaluator, while the finding's nominal metric type selects the strategy or
builder that owns any executable refactor concept.

The generated :doc:`detector_catalog` derives each detector's required-relation
owner, contribution contracts, C3 resolution paths, recipe-evaluation
capability, recipe-synthesis capability, and refactor concept directly from
those declarations. Authority-boundary evidence identifies the boundary and
does not imply that its authority was recovered. Likewise, recipe-synthesis
capability is not evidence that a detector contributed a valid refactor for a
particular repository. Source-specific authority evidence belongs to the
semantic refactor gate, while evaluation, proof obstacles, executable recipes,
and planning horizon belong to finding recipe synthesis. Finding metrics supply
typed evidence to a declared evaluator; they do not grant recipe capability.
Run ``nominal-refactor-advisor --detector-capabilities`` to emit the same
declaration-derived inventory as JSON without performing a source scan.

Contract-member evidence derives required slots from the contract's native
``__abstractmethods__`` set, including inherited obligations. Implementation
ownership is the first declaration supplying the member in Python's MRO. An
abstract override cannot be bypassed to claim a concrete ancestor's
implementation, and virtual ABC registration does not establish native
derivation. The recorded :download:`native contract evidence refactor
<../../examples/native_contract_evidence_refactor.py>` applies this change
through scoped assignments, a function prelude and exact target patches.

The ``repeated_local_regex_bundle`` observer recognises calls through stable
module bindings to declared standard-library regex operations. Lexical lookup
excludes parameters and local names that shadow those bindings. Module aliases,
imported function aliases and single-assignment captured aliases are supported;
argument positions and keywords are checked against each operation's native
signature. Unpacked calls, invalid bindings and unresolved local aliases do not
contribute pattern evidence. This is a source-level observation, not a runtime
equivalence or pattern-validity proof. Ownership remains undecided, and the
observer does not synthesise a rewrite or prescribe a new grammar class.

The observer uses the shared function syntax index, lexical dependency
projection, declaration index and module binding snapshot. Modules without
relevant imported bindings skip the deeper lexical analysis. A bounded source
projection cache reuses these facts across scans; threshold-dependent decisions
remain uncached. ``release_module_analysis_memory`` clears that cache through
the existing discovery mechanism. The recorded
:download:`regex observation cleanup <../../examples/regex_observation_cleanup.py>`
removes the obsolete spelling-based collector and duplicate function index;
the new source-backed detector declaration is a prerequisite of that plan.

An authority-producing detector emits its exact source witness through
``RefactorFinding.authority_evidence``.  The witness must also belong to the
finding's evidence tuple, so finding-backed semantic-descent graphs are derived
from the finding record itself.  Graph construction does not rejoin detector
identities to a separate evidence-position table.


Structural Hypothesis Substrate
-------------------------------

Structural pattern metadata is carried by ``PatternId`` members in
:mod:`nominal_refactor_advisor.patterns`. The planner module clusters findings
through shared source evidence without selecting an application order.

For maintainers, the important split is:

- each ``PatternId`` member owns its broad pattern identity, descriptive required
  relation, and witness capabilities
- each detector derives its executable obligation identity from the class in its
  MRO that physically declares the selected ``finding_spec``
- ``build_refactor_plans`` preserves every observed pattern in stable identity order
- pattern evidence states the missing relation without prescribing a normal form
- graph execution classes expose structural evidence only; they do not rank or
  schedule refactors

The public ``build_refactor_plans`` entrypoint is documented in
:doc:`public_api`.
