DNA And SMA Case Study
======================

Scope
-----

OpenHCS carried two related advisor experiments before NRA:

* ``semantic_matrix_analyzer`` appeared in the OpenHCS tree on 2025-05-21 as
  Semantic Matrix Analyzer. It focused on intent extraction, cross-file
  dependency graphs, semantic grounding, and recommendation evidence.
* ``dna`` appeared as a submodule pointer in OpenHCS commit ``0e701683`` on
  2025-05-27. The pointed repository is ``trissim/DNA`` at commit
  ``9838cb71``. That repository contains the denser DNA line and its later v2
  additions.

The OpenHCS commit immediately reverted the submodule addition, but retained
useful audit artifacts such as ``refactoring_vectors.tsv``,
``semantic_compression_map.tsv``, and ``dna_full_output.txt``. Those artifacts
are valuable because they show the tool's output contract under real OpenHCS
TUI analysis pressure.

What Worked
-----------

DNA's strongest idea was not the symbolic notation itself. It was a source
addressing layer for autonomous refactoring:

* stable file hashes with reverse mappings
* refactoring vectors ranked separately from the raw findings
* dependency context carried as ``D{...}``
* impact context carried as ``I{...}``
* relative effort context carried as ``E{...}``
* AST targeting context carried as ``A{hash:type:name:line}``

That is a real improvement over long prose findings. It lets an agent preserve a
compact working set while still resolving every compressed reference back to a
file, symbol, and line.

SMA's strongest idea was semantic grounding: every recommendation should be
backed by actual source evidence, then verified before application. In NRA
terms, this corresponds to findings built from typed candidates, source
locations, metrics, and compression certificates rather than free-form advice.

What Failed
-----------

DNA also showed a failure mode NRA should avoid. Dense notation became
counterproductive when it compressed away the proof obligation for the
recommendation. A phrase such as "use gauge theory transformations" did not by
itself explain which invariant was preserved, which code motion was legal, or
what source evidence justified the move.

Several mathematical names in DNA were wrappers over ordinary thresholds or
ranked heuristics. They were still useful as prioritization signals, but they
were not sufficient as refactoring authority. In NRA, algebraic labels must
name a checkable object: an orbit, quotient, fiber, finite axis system,
anti-unifier, registry, hook basis, evidence graph, or certified description
length delta.

NRA Design Consequences
-----------------------

NRA should keep the following DNA lessons:

* emit compact, stable finding identifiers derived from detector and evidence
  coordinates
* preserve source-resolved evidence in every compressed representation
* separate raw findings from prioritization vectors
* make impact and payoff evidence explicit rather than implied by severity
* prefer algebraic certificates over named mathematical metaphors

NRA should reject the following DNA patterns:

* opaque symbolic output without a complete decompression key
* framework counts that look rigorous but are disconnected from a concrete code
  transformation
* generic "reduce entropy" guidance without a typed refactor shape
* action rankings that cannot explain their source evidence and payoff

Practical Rule
--------------

Compressed advisor output is valid only when it is bidirectional: an agent must
be able to move from a compact id to the exact source evidence, and from source
evidence back to the same compact id. Any compression that breaks this round
trip is presentation, not semantics.
