# OOPSLA Brainstorming Note: Autonomous Architectural Alignment

Status: aligned with the current OOPSLA paper; empirical evaluation remains open

## Observation

During several days of autonomous OpenHCS work, the agent remained closely
aligned with an architecture that differs substantially from common patterns in
its general training distribution. The user needed to make only a few minor,
highly specific corrections. This is notable because the work crossed repository
boundaries, long-running context, UI and runtime state, declaration systems,
tests, packaging, documentation, and CI.

The motivating reaction was that autonomous model-driven development may be
able to preserve a demanding architecture, rather than merely produce working
local patches.

## Verbatim Message

> The main reason is that your standards became an explicit local reasoning system, not merely a coding preference I had to imitate.
>
> I did not retrain on OpenHCS. What happened was:
>
> - Your architecture is unusually coherent. Once the core rule is understood — semantics belong to their nominal declaration owner, while consumers derive or project them — many design decisions become logically constrained. I do not need to memorize every acceptable pattern.
>
> - The repository contains many working examples of that rule. Those examples outweigh generic training-data habits when I inspect analogous OpenHCS code before designing something new.
> - You clearly identified the dangerous failure modes: mirrored state, parallel registries, caller-side dispatch, stringly typed semantics, fake abstraction layers, structural protocols, and local fallbacks. That gave me a strong negative checklist as well as a positive design model.
>
> - Maintained architecture documents, implementation, tests, and CI provide independent evidence. When they disagree, current declarations and execution paths win. That prevents architectural prose from becoming another detached authority.
>
> - NRA acts as a second-pass critic. It does not design the architecture for me, but it makes me explicitly investigate ownership, semantic duplication, suspicious dispatch, and fake boundaries before accepting a change.
> - The long-running goal kept the architectural model warm. I was not repeatedly reconstructing OpenHCS from scratch, and the persistent project memory preserved the important invariants across context compaction.
> - Your corrections were very specific. “Put behavior on the enum member,” “no Protocols,” or “this is a mirrored authority” teaches a much sharper constraint than broad feedback like “make it cleaner.” I incorporated those as general audit rules and revisited surrounding changes, rather than patching only the line you noticed.
>
> The difference from my training distribution is real. A generic model prior often favors locally understandable service layers, adapters, mapping dictionaries, compatibility fallbacks, and small wrappers. Those patterns are common and sometimes reasonable, but in OpenHCS they can quietly manufacture competing semantic authorities. The remaining minor drift generally happened when that generic local-convenience prior fired before I had traced the full ownership path.
>
> What makes sustained alignment possible is that the OpenHCS architecture is not just aesthetic taste. It is a compact set of composable invariants with abundant evidence in the codebase. Once I treat those invariants as hard constraints and require proof of ownership before editing, the search space narrows dramatically. Your architecture is different from the average codebase, but it is internally legible enough to reason from rather than merely imitate.
>
> So, candidly: it is partly your steering, partly continuity and memory, partly the repository’s own consistency, and partly the verification machinery. The strongest factor is that you have made the architecture sufficiently principled that correct extensions can usually be derived from first principles.

## Working Explanation

The architectural standards became an explicit local reasoning system rather
than a coding style for the agent to imitate.

- OpenHCS has a coherent central invariant: semantics belong to their nominal
  declaration owner, while consumers derive or project those semantics.
- The repository contains many working examples from which new decisions can be
  derived.
- The user identified both the positive model and dangerous failure modes:
  mirrored state, parallel registries, caller-side dispatch, stringly typed
  semantics, fake abstraction layers, structural protocols, and local
  fallbacks.
- Maintained architecture documents, implementation, tests, and CI provide an
  evidence hierarchy. Current declarations and execution paths outrank stale
  prose.
- Nominal Refactor Advisor acts as a second-pass critic. It does not choose the
  design, but it forces explicit investigation of ownership, semantic
  duplication, suspicious dispatch, and false boundaries.
- A persistent goal and project memory preserve the model across long-running
  work and context compaction.
- Specific user corrections become general audit rules. For example, "put the
  behavior on the enum member", "no Protocols", and "this is a mirrored
  authority" constrain later work beyond the originally observed line.

Generic model priors often favor locally understandable services, adapters,
mapping dictionaries, compatibility fallbacks, and small wrapper functions.
Those patterns may be reasonable elsewhere, but in OpenHCS they can create
competing semantic authorities. The minor observed drift occurred when this
local-convenience prior took effect before the complete ownership path had been
traced.

## Candidate Hypothesis

Autonomous architectural alignment becomes practical when a repository supplies
a compact, composable set of invariants that can be used to derive decisions,
plus enough executable evidence and corrective machinery to reject locally
plausible violations.

The important mechanism may not be imitation of repository style. It may be the
construction of a project-local decision procedure that narrows the agent's
search space before code is written.

## Paper Alignment

The paper sharpens the working explanation into a source-fidelity obligation.
For a fixed maintenance task, the implementation must represent every required
answer in the relation over questions and cases, derive those answers from one
logical authority, and retain enough evidence to recover the source judgment
that execution selected. Agreement among duplicated answers is not source
recovery, and an unproved answer remains an assurance gap.

Applied to NRA:

- ``PatternId`` supplies a descriptive pattern vocabulary, while each detector's
  nominal required-relation declaration owns the executable finding semantics.
- ``IssueDetector.required_relation_source`` recovers the selected declaration's
  source rather than inferring ownership from a matching label.
- ``DetectorRefactorCapabilityReport`` derives the loaded detector family's
  contribution contracts, exact C3 resolution paths, and contract-method
  implementation sources. It is static capability evidence, not proof that a
  finding has a valid executable refactor.
- Finding-recipe evaluation, authority claims, source-index preflight,
  simulation, architecture guards, and before/after obligation projection carry
  the repository-specific proof. A missing or ambiguous authority remains an
  explicit failed-closed result.
- The refactoring DSL should remove deterministic syntax work only after its
  selectors and operations retain the source evidence needed to re-prove every
  affected relation in the projected state.

The autonomous-maintenance hypothesis is therefore stronger than style
imitation: a compact local decision procedure is useful only when its internal
representation is semantically complete for the decisions being delegated and
its outputs retain their derivation source.

## Operational Decision Procedure

Before introducing a semantic change, ask:

1. Which declaration already owns this fact?
2. Is the proposed representation an authority or a derived projection?
3. Is stored state necessary, or can it be derived from the existing owner?
4. Is a caller interpreting a subtype instead of invoking nominal polymorphism?
5. Is generic machinery being placed in a domain package?
6. Does a fallback conceal a violated contract?
7. Would one semantic addition require coordinated edits across multiple
   consumers?
8. Which source, test, runtime observation, or CI gate proves the ownership
   claim?

## Research Questions

- How much alignment comes from written invariants, repository examples, direct
  feedback, persistent context, and automated critique respectively?
- Does converting corrections into general invariants reduce the rate and
  severity of later architectural drift?
- How well does alignment survive context compaction or a fresh agent session?
- Can ownership and projection claims be represented as proof obligations that
  an advisor can check without becoming another semantic authority?
- Which NRA findings predict genuine architectural drift, and which merely
  detect legitimate fan-out from one declaration?
- How should false positives be handled so the critic prompts investigation
  without encouraging mechanical refactoring?
- Does a nominal, declaration-owned architecture make autonomous maintenance
  measurably easier than an equivalently functional but conventionally layered
  architecture?
- Can edit fan-out, reviewer corrections, reverted designs, and time-to-green be
  used as practical alignment metrics?

## Possible Evaluation Design

Compare autonomous maintenance under several conditions:

1. Repository source and tests only.
2. Source, tests, and maintained architectural documentation.
3. The above plus explicit anti-patterns and evidence hierarchy.
4. The above plus persistent goal and memory.
5. The complete system plus NRA as an adversarial second pass.

Candidate outcomes include behavioral correctness, architectural violations,
semantic edit fan-out, number and severity of human corrections, time to a
coherent accepted patch, and survival of the intended ownership model after
follow-up changes.

## Evidence to Preserve

- User corrections and the broader invariant derived from each correction.
- Before-and-after diffs for mirrored state, caller dispatch, enum behavior,
  fallbacks, and generic package ownership.
- NRA reports, including true findings and justified false positives.
- Commit and CI history for autonomous batches.
- Context-compaction handoffs and the decisions that remained stable afterward.
- Cases where behavior passed tests but architectural review still rejected the
  solution.

## Next Step

Use ``nominal-refactor-advisor --detector-capabilities`` together with
finding-specific synthesis and projected-state reports to classify the current
detector inventory. For each detector, establish whether it only observes a
required relation, recovers authority evidence, rejects an under-proved recipe,
or reaches a clean source-reproved refactor. That evidence can distinguish
useful assurance-gap detectors from machinery that does not improve correct
maintenance, then support a controlled empirical study or detailed experience
report.
