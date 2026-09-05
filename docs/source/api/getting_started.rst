Getting Started
===============

The advisor is an AST-driven tool for finding structural refactors that collapse
duplicate authority, replace duck-typed branching with nominal boundaries, and
derive repeated surfaces from one canonical source.

Quickstart
----------

Analyze a package from the CLI:

.. code-block:: bash

   nominal-refactor-advisor path/to/python/package

Emit JSON instead of Markdown:

.. code-block:: bash

   nominal-refactor-advisor path/to/python/package --json

Include graph-clustered structural hypotheses:

.. code-block:: bash

   nominal-refactor-advisor path/to/python/package --include-plans

Focused Edit Loops
------------------

For a bounded file-focused check during development, use the compact loop
payload:

.. code-block:: bash

   nominal-refactor-advisor changed_module.py --json --json-payload loop

When this command infers a larger package context and no reusable context cache
exists, it analyzes the requested files with per-module detectors only.  The
JSON ``scan_status`` then reports ``focused_local_partial`` together with the
analyzed and omitted detector counts.  This is an intentionally incomplete
edit-loop result, not proof that context-dependent findings are absent.

Use the full payload or provide ``--context-root`` when an exact contextual
scan is required:

.. code-block:: bash

   nominal-refactor-advisor changed_module.py --context-root path/to/package \
     --json --json-payload full

Inspecting Bounded Codemod Candidates
-------------------------------------

To inspect a declaration-owned recipe against the current source snapshot,
request a simulation:

.. code-block:: bash

   nominal-refactor-advisor path/to/python/package \
     --codemod-synthesize-plan --codemod-simulate --json

This proves only that the candidate is coherent in the current snapshot.  It
does not establish that the candidate belongs to a globally complete refactor
trajectory.  NRA therefore blocks one-shot plan export and source application
while the planning horizon is ``current_snapshot`` or ``unproved``.

Some findings intentionally have no recipe.  Repeated source proves that one
maintenance object exists, but it does not necessarily prove where that object
belongs.  NRA keeps such findings as evidence instead of inventing an authority.

Applying an Ordered Hand Patch
------------------------------

When you already know the exact source transformations, use ``patch_target``
to apply them to one indexed target in order:

.. code-block:: bash

   nominal-refactor-advisor path/to/python/package \
     --codemod-plan - --codemod-simulate <<'JSON'
   {
     "recipes": [{
       "recipe_id": "replace-legacy-rendering",
       "operations": [{
         "operation": "patch_target",
         "file_path": "package/rendering.py",
         "target_qualname": "Renderer.render",
         "replacements": [
           {
             "old_source": "legacy(value)",
             "new_source": "prepared(value)"
           },
           {
             "old_source": "prepared(value)",
             "new_source": "Renderer.prepare(value)"
           }
         ]
       }]
     }]
   }
   JSON

Each replacement sees the result of the preceding replacement.  NRA requires
each old source fragment to occur exactly once, compiles the chain into one
physical rewrite, validates the resulting Python, and writes nothing when any
step fails.  Review the simulated diff, then use ``--codemod-apply``.

Chaining a Hand Patch into a Semantic Refactor
----------------------------------------------

Use separate ``stages`` when a semantic operation must resolve against the
result of a hand patch:

.. code-block:: bash

   nominal-refactor-advisor path/to/python/package \
     --codemod-plan - --codemod-simulate <<'JSON'
   {
     "stages": [
       {
         "recipes": [{
           "recipe_id": "bind-helper-to-shared-context",
           "operations": [{
             "operation": "patch_target",
             "file_path": "package/source.py",
             "target_qualname": "Helper.resolve",
             "replacements": [{
               "old_source": "LegacyContext) -> LegacyContext:",
               "new_source": "SharedContext) -> SharedContext:"
             }]
           }]
         }]
       },
       {
         "recipes": [{
           "recipe_id": "extract-patched-helper",
           "operations": [{
             "operation": "extract_symbols_to_new_module",
             "file_path": "package/source.py",
             "symbol_qualnames": ["Helper"],
             "destination_path": "package/helper.py"
           }]
         }]
       }
     ]
   }
   JSON

NRA reindexes the projected source at the stage boundary.  The extraction
therefore selects the patched declaration and derives its import and consumer
rewrites from that state.  Review the combined diff, then replace
``--codemod-simulate`` with ``--codemod-apply`` to write the sequence as one
revision-checked batch.

Hand-authored source for ``insert_before_target`` and ``insert_after_target``
contains the declaration itself, without leading or trailing padding.  The
operation derives module-level or nested spacing from the selected target.
When multiple operations insert at the same anchor, each inserted declaration
owns its typed leading boundary, so generated imports and moved declarations
compose without a formatter cleanup stage.

Use ``erase_dead_compatibility`` when deleting an obsolete declaration must
also prove that named calls or attributes do not survive anywhere in the
repository.  Its residual-use guard is part of the registered operation and is
therefore available in JSON plans and ordered sequences; it is not a separate
Python-only cleanup helper.

Use ``delete_module_call_declarations`` for module-level declarations produced
by a factory call rather than a class or function statement.  The operation
selects calls by their qualified callee and a positional-argument name prefix,
requires an explicit cardinality when ambiguity matters, and derives deletion
geometry from the current parsed module.  The plan names the semantic factory
relation without copying source spans or resorting to a textual patch.

.. code-block:: json

   {
     "operation": "delete_module_call_declarations",
     "file_path": "package/rules.py",
     "callee_qualname": "declare_rule",
     "positional_argument_prefix": ["LegacyRuleCandidate"],
     "selection_count": {"exact": 1}
   }

Extracting Methods into an Ancestor
-----------------------------------

To move related methods into a shared ancestor, compose the declaration,
inheritance, and member-promotion operations in successive stages. Each stage
resolves targets against the source produced by the previous stage.

The following plan extracts two renderer helpers from NRA's historical
``codemod.py`` at revision ``b849d95``. The final stage rewrites the promoted
method's multiline signature, retaining its parameters and annotations:

Use ``CodemodPlanSequence.from_operations`` to write the edits directly in Python.
It derives the recipe wrappers and stage identifiers and re-proves each operation
against the preceding stage's output. Reuse ``SourceRewriteTarget`` objects when
several edits address the same declaration:

.. literalinclude:: ../../examples/renderer_refactor.py
   :language: python
   :start-at: module =
   :end-before: WITNESS =

Download the :download:`complete Python example <../../examples/renderer_refactor.py>`
and change its file, class, and member names for your source. The script emits
the normal JSON plan; it does not apply edits. Preview its combined extraction
and witness migration:

.. code-block:: bash

   python renderer_refactor.py | nominal-refactor-advisor path/to/package \
     --codemod-plan - --codemod-simulate

Review the diff, then use ``--codemod-apply`` in place of
``--codemod-simulate`` and run the affected tests. Method promotion moves the
existing bodies and decorators; the plan does not contain copies of them.

Applying a stored simulation rechecks every file in its supplied source snapshot,
including files used only for analysis. If a recorded source changed, simulate
again before applying. Only files in the write set are modified. Rescan the
repository after adding source files so the snapshot includes the new declarations.

Use ``CodemodPlanDocument`` when multiple operations must resolve against the
same snapshot, then combine documents or sequences with ``CodemodPlanSequence.compose``.
``from_operations`` intentionally gives each operation its own projected state.

To replace one direct base while retaining the declared base order, use
``ReplaceClassBaseOperation``:

.. code-block:: python

   ReplaceClassBaseOperation(
       target=SourceRewriteTarget(file_path="package/worker.py", qualname="Worker"),
       base_name="LegacyContext",
       replacement_base_name="SharedContext",
   )

Declare or import the replacement first. The operation preserves the other bases,
class keywords, generic parameters and body. Check MRO-dependent behaviour in the
affected tests. Use ``replace_direct_class_base`` when the intended scope is an
authority's complete direct-child cohort instead of one selected class.

``replace_function_signature`` accepts a single-line replacement suffix for
either a single-line or multiline original signature. It retains the function
name, generic type parameters, decorators, body, and comments outside the
replaced span. Comments inside that span require an explicit edit before
replacement. Changes to parameter names or calling conventions need their own
body and caller edits in the plan.

Passing Source Evidence Through a Witness
-----------------------------------------

Once you have chosen a context object for a function, add its parameter with
``replace_function_signature``. Use ``project_function_parameter`` to redirect
reads of an existing parameter to an access path such as ``witness.candidate``.
Both the original parameter and the access path's root must exist at this stage.
Then remove the old parameters and update the callers in the same batch.

The ``WITNESS`` sequence in the Python example extends the extraction above.
Its ``PLAN`` composes both sequences with ``CodemodPlanSequence.compose``.
The regression test obtains its witness through NRA's actual source-reproof
operation and executes the renderer helpers before and after applying the plan.

Parameter projection checks lexical ownership. It preserves shadowed bindings,
ordinary string literals, and comments, and rejects rebinding or capture of the
projection root. The author chooses the field relationship; the operation does
not infer it or change signatures and callers automatically. Review reflected
parameter names and debug/template-string expression labels when changing an API.

Use ``prepend_function_body`` to introduce statements before the existing
executable body. It preserves the docstring and existing statements, including
nested decorators, and expands inline suites when needed. The witness example
uses this operation to introduce the new call derivation.

After changing the signature, use ``replace_declared_call_arguments`` in the
next stage to update calls in a selected scope. Its ``callee`` selector names the
declaring function or method, including calls through an inheriting class. Set
``arguments_source`` to the new argument list and ``selection_count`` to the
expected number of calls. The operation resolves the callee again and checks the
new arguments against its current signature; unrelated same-named methods are
left alone. It rejects unresolved selections, argument unpacking, and edits that
would discard argument comments.

You choose the new expressions and their evaluation order. Signature binding is
not a proof that those expressions preserve behaviour.

To replace a helper call with a shared method or an attribute access, use
``replace_declared_call`` with the same ``target``, ``callee``, and
``selection_count`` selectors. Set ``expression_source`` to the replacement,
for example ``insertion_point.member_insertion_replacement(member_sources)``.
The operation replaces each selected call as a complete expression, retaining
the surrounding expression's precedence. Calls to unrelated declarations stay
unchanged; unresolved selections and removal of existing comments are rejected.

Choose this operation when the replacement changes the callee or removes the
call altogether. Its checks establish the selected declaration and valid Python
syntax, not equivalence of the authored expression or binding of a new callee.
Review evaluation effects and run the relevant behavioural tests. Follow it
with assignment deletion and import removal in the same sequence when those
declarations become unused.

Use ``delete_function_assignments`` to explicitly remove the old direct
assignments by name. This removes their evaluations too, including any calls
and attribute access. Choose this step only after reviewing those effects and
the remaining uses of the names. Repeated assignments to the same name,
partially selected chained bindings, and mixed attribute or subscript writes
are rejected. The operation preserves neighbouring statements on the same line
and supplies ``pass`` when the function would otherwise have an empty body.
It also keeps an ordinary string expression from becoming a new docstring.

The composed renderer example now uses declaration-selected operations
throughout, with no exact-text patches.

Reusing an Existing Authority
-----------------------------

To change a function's decorators without rewriting its implementation, use
``replace_function_decorators`` with a declaration ``target`` and
``decorators_source``, such as ``"@cached_property"``. The source can contain
multiple decorators in their intended order; an empty string removes them.
Existing comments in the replaced block require review and are not discarded.

This recorded NRA plan memoizes three projections of an immutable declaration:

.. literalinclude:: ../../examples/cache_call_declaration_projections.py
   :language: python

Review dependency mutability and decorator behaviour before applying such a
change. The operation checks Python syntax and source ownership, not whether
memoization is appropriate for the selected function. Its header and body remain
unchanged, including comments and multiline literals.

To share an existing function implementation, use ``alias_function`` with a
``target`` and an ``implementation`` selector. Both declarations must be in the
same lexical scope, and the selected implementation must be bound before the
target. The operation replaces the target declaration with a named assignment
and rejects edits that would discard comments.

This plan consolidates NRA's import visitors, widening the shared signature
before introducing the alias:

.. literalinclude:: ../../examples/import_visitor_refactor.py
   :language: python

Aliasing shares the implementation's function object, including its name,
annotations, defaults and descriptor binding. It removes evaluation of the old
definition's decorators, defaults and annotations. Review those effects and
behavioural equivalence before applying the plan; its checks establish binding
availability and source ownership, not equivalent behaviour.

To edit calls after introducing an alias, append ``replace_declared_call`` or
``replace_declared_call_arguments`` to the sequence. Select the original
``implementation`` declaration as the callee. The next stage follows the alias
in the projected source, including same-class method aliases, so the plan does
not need to rediscover or text-match the new call spelling.

Use ``project_function_local`` to replace reads of a single-assignment local
with an existing parameter's access path, such as ``self.geometry``. Set
``local_name`` to the local binding and ``projection_source`` to the access
path. The operation retains the initialiser and its effects; remove it with
``delete_function_assignments`` in the next stage when it is no longer needed.

This executable plan records NRA's own migration to its existing geometry
authority:

.. literalinclude:: ../../examples/source_geometry_refactor.py
   :language: python

Local projection uses the same lexical ownership and capture checks as
parameter projection. It requires one direct, single-name assignment with a
value, rejects other writes to that binding, and rejects reads appearing before
the initialiser completes. Shadowed names in nested scopes remain unchanged.

The author chooses the value relationship. Accessing a property repeatedly can
differ from retaining a local snapshot, so review its lifetime and side effects
before making this change. In the geometry example, the destination is an
existing cached property over the same immutable source buffer.

Converting a Detector to a Declaration
--------------------------------------

To replace a direct finding-builder method and then its metadata-only detector
class, put the two transformations in separate stages. Change the file and
target names in this plan to match your detector:

.. literalinclude:: ../../examples/detector_declaration_sequence.json
   :language: json

Save the plan as ``detector-plan.json`` and preview the combined change:

.. code-block:: bash

   nominal-refactor-advisor path/to/package \
     --codemod-plan detector-plan.json --codemod-simulate

The second stage reads the renderer declaration produced by the first. Each
operation checks the current collector base, source shape, and relevant scope
dependencies. The plan stores targets, not replacement text. Review the diff,
then replace ``--codemod-simulate`` with ``--codemod-apply`` to apply the batch.
Run your detector tests against the result.

Renaming a Top-Level Binding Authority
--------------------------------------

Use ``rename_top_level_binding_authority`` when the authority is declared by a
module assignment rather than a class or function definition:

.. code-block:: json

   {
     "operation": "rename_top_level_binding_authority",
     "file_path": "package/monolith.py",
     "binding_name": "_LegacyNode",
     "new_name": "AstNode"
   }

The operation proves one unambiguous assignment declaration, then derives its
repository imports, exports, direct references, qualified references, and
annotations.  Put a subsequent deletion, import, or extraction in the next
stage when it must resolve the renamed binding.

Extracting a Declaration Closure
--------------------------------

To extract a declaration and its movable source-local dependency closure into a
new module, provide only the semantic roots:

.. code-block:: bash

   nominal-refactor-advisor path/to/python/package \
     --codemod-plan - --codemod-simulate <<'JSON'
   {
     "recipes": [{
       "recipe_id": "extract-source-edit-algebra",
       "operations": [{
         "operation": "extract_symbol_closure_to_new_module",
         "file_path": "package/monolith.py",
         "root_symbol_qualnames": ["NominalSourceEdit"],
         "maximum_moved_symbol_count": 12,
         "destination_path": "package/source_edits.py"
       }]
     }]
   }
   JSON

Review the simulated diff, then replace ``--codemod-simulate`` with
``--codemod-apply``.  NRA derives transitive movable declarations, imports,
source re-exports, repository-local consumer imports, and the new-file revision
contract.  A non-movable or unresolved dependency fails preflight without
writing either module.
The dependency report preserves both ``requested_symbol_names`` and
``moved_symbol_names`` and derives the added closure as
``derived_symbol_names``.  Review that distinction before applying a move; a
large derived closure is evidence that the selected declaration still depends
on a broader authority boundary.  Closure moves require an explicit
``maximum_moved_symbol_count`` and fail preflight if the derived selection
exceeds that bound.

For a multi-step extraction, the next operation can select a module created by
the preceding one. This recorded NRA refactor first extracts signature binding,
then separates its value-expression dependencies:

.. literalinclude:: ../../examples/call_binding_extraction.py
   :language: python

The plan names two semantic roots. It derives the remaining declarations and
updates consumers after each stage; no declaration bodies or consumer lists are
copied into the plan. This extraction is already applied in NRA. Its CLI
regression exercises the recorded plan against a pre-extraction fixture.

Runtime imports remain runtime imports.  Imports declared under a recognised
``typing.TYPE_CHECKING`` guard retain that scope when the dependency is used
only by deferred annotations.  NRA also rewrites guarded imports in repository
consumers and fails preflight when eager annotation evaluation would make a
guarded dependency unavailable.

Relative imports are resolved to their canonical module identity before a
declaration moves.  NRA then renders the dependency from the destination, so a
move into a deeper or shallower package does not silently retarget the import.
An existing destination binding satisfies the dependency only when its resolved
authority matches; a same-named local declaration or different import fails
preflight.
Import aliases are presentation derived from the required bound name; redundant
same-name aliases do not create a second import authority.

Source and existing destination modules must use the same annotation evaluation
policy when moved declarations contain annotations.  New-module extraction
derives the source module's policy when ``destination_source`` is omitted,
including ``from __future__ import annotations`` when required.  An explicitly
supplied destination source remains authoritative; NRA fails preflight if it
would change annotation semantics.

To prove a goal across reachable source states, run:

.. code-block:: bash

   nominal-refactor-advisor path/to/python/package \
     --codemod-refactor-goal semantic_carrier --json

The goal runner enumerates every clean compatible recipe batch at each state,
explores the resulting exact source-state graph, and deduplicates cycles by
source identity.  It emits an applicable replay sequence only when complete
exploration reaches one unique terminal source state.  Local compression scores
and candidate order do not select a branch.

Guards supplied through ``--codemod-plan`` are terminal invariants.  The search
may cross an intermediate state that violates them when a later stage repairs
that state.  The terminal source must satisfy every supplied guard, and the
exported replay attaches those guards to its final stage.  Guards owned by an
individual recipe continue to validate that recipe's immediate result.

``--codemod-goal-max-stages``, ``--codemod-goal-max-states``, and
``--codemod-goal-max-branches`` bound the proof search.  Reaching any bound
produces an ``incomplete`` trajectory proof and leaves the checkout unchanged.
Distinct terminal states produce ``ambiguous_terminal_states``; a completely
explored graph with no terminal produces ``no_terminal_state``.

The default persistent cache is maintained at most once per hour.  Retention
keeps at most 128 recently used analysis roots, 4 GiB across those roots, 2 GiB
within one active root, and four recent exact semantic-graph generations.
Explicit ``--cache-dir`` locations are caller-managed and are not pruned by
this policy.

Reproducible Performance Gate
-----------------------------

``nominal-refactor-benchmark`` runs the compact focused scan twice against one
isolated cache, samples the complete process-tree RSS, and emits cold/warm JSON
measurements.  Optional ceilings turn it into a regression gate:

.. code-block:: bash

   nominal-refactor-benchmark \
     --max-cold-seconds 15 --max-cold-rss-mb 180 \
     --max-warm-seconds 5 --max-warm-rss-mb 180 \
     changed_module.py another_changed_module.py

The command fails when either subprocess times out, leaks a nonzero exit,
emits invalid JSON, changes finding counts between cold and warm runs, omits
the partial-scan contract, or exceeds a supplied time/RSS ceiling.

What Stays Stable
-----------------

For downstream use, treat these as the main supported surfaces:

- ``analyze_path`` for finding generation
- ``plan_path`` and ``build_refactor_plans`` for non-actionable structural hypotheses
- ``AnalysisReport``, ``RefactorFinding``, and ``RefactorPlan`` for results
- ``PatternId`` for canonical pattern identity and metadata

How To Read The Rest Of The Docs
--------------------------------

- Use :doc:`public_api` for importable entrypoints.
- Use :doc:`theory_and_results` for result dataclasses, taxonomy, and pattern metadata.
- Use :doc:`pattern_catalog` and :doc:`detector_catalog` for the current shipped behavior.
- Use :doc:`../development/index` for rationale, case studies, and maintenance workflow.

Architecture Map
----------------

- ``nominal_refactor_advisor.cli``: CLI entrypoints and output formatting
- ``nominal_refactor_advisor.detectors``: registered detector family and finding synthesis
- ``nominal_refactor_advisor.patterns``: pattern metadata shared by findings, hypotheses, and docs
- ``nominal_refactor_advisor.models``: result records and finding metrics
- ``nominal_refactor_advisor.observation_*``: structural observation substrate
- ``nominal_refactor_advisor.planner``: subsystem-level evidence clustering over findings

Building Docs
-------------

.. code-block:: bash

   pip install -e .[docs]
   python -m sphinx -b html docs/source docs/_build/html
