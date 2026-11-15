Leveraging NRA Scans for Batch Refactoring
===========================================

Operational patterns for turning NRA scan results into large batch folds.

The Scan Is the Map
-------------------

One complete scan (``python -m nominal_refactor_advisor --json``) returns every
finding the detectors prove.  Save it once::

    python -m nominal_refactor_advisor --json 2>/dev/null > findings.json

Query the saved dump instead of rescanning per edit.  Rescan when the class
graph changes (new base classes, moved fields, deleted types), not after
consumer-only edits.

The ``--include-structural-overlap`` flag adds overlap groups in about 2 seconds
on a 274-file codebase.  Use it to see which findings cluster.

Codemod Synthesis
-----------------

``--codemod-synthesize-class-plan --json`` runs the full analysis and returns
executable candidates the planner can prove::

    python -m nominal_refactor_advisor --codemod-synthesize-class-plan --json \
        2>/dev/null > synthesis.json

Read ``synthesis.json`` for:

- ``finding_recipe_plan.synthesis_report.records``: every finding with its
  synthesis status (``executable_candidate``, ``no_evaluator``,
  ``rejected_by_safety_check``)
- ``finding_recipe_plan.trajectory_frontier.branches``: alternative refactor
  trajectories
- ``finding_recipe_plan.application_blocked`` / ``application_block_reason``:
  whether the planner can auto-apply

The plan application (``--codemod-plan-out``) may fail on source-index gaps
(e.g., ``__init__.py`` unavailable).  In that case, apply the proven candidates
by hand using the synthesis report as the guide.

``--search-refactors`` explores deeper but uses proportionally more memory.
On a 274-file codebase it OOMs above ``--search-width 3``.  Use synthesis
instead for the executable frontier.


Detector Leverage Tiers
-----------------------

Not all 1500 findings are equally actionable.  Batch by detector:

**Tier 1 — mechanical, batch now**

``foreign_family_routable_filter``
    Sites testing ``isinstance(node, (A, B))`` where a ``ForeignFamily``
    already registers both classes.  Replace with
    ``isinstance(node, Family)`` or ``Family.within(nodes)``.  One import
    plus one token change per site.  The 2026-11-15 session converted 91
    such sites across 49 files in one commit.

``foreign_adapter_bind_chain``
    Sites decoding an AST node class-by-class where a ``ForeignAdapter``
    already provides the unified interface.  Replace the isinstance block
    with ``Adapter.within(statements)`` and read ``.targets``, ``.value``,
    etc.  The 2026-11-15 session converted 4 sites from 10-line blocks to
    4-line iterations.

``linear_scan_by_name``
    ``next(x for x in collection if x.name == target)`` where a dict
    keyed by name is O(1).  Add a ``@cached_property`` dict and replace
    the scan.

``missing_mixin`` (dataclass-to-dataclass only)
    Classes that share 2+ field declarations with another class.  Extract a
    shared base declaring those fields; each class inherits it.  This is
    blocked when callers construct positionally — convert to keyword first.

**Tier 2 — per-site design, high leverage**

``own_type_test``
    ``isinstance(x, OwnClass)`` in consumers of a family.  The fix is a
    virtual method on the base, overridden per member.  The largest
    cluster (``OpenCapturedReference``, 18 sites) requires understanding
    what each site does after the check.

``phantom_family``
    Two+ classes declare the same method with no shared ancestor.  The fix
    is an ABC declaring the method.  Group by class pair: a pair sharing 7
    methods needs one ABC, not 7 single-method extractions.

``forwarding_facade``
    A class holds another and forwards N properties.  Fix by MI (inherit
    the held class) or by deleting the forwarding and changing consumers.
    Check for field-name collisions before MI.

**Tier 3 — small or external, skip unless in the path**

``foreign_absence_probe``
    Probing external AST objects for ``None`` (e.g., ``keyword.arg is None``).
    Usually a boundary check against a foreign type, not a missing family.

``string_kind_dispatch``
    Comparing against string sets that are external vocabularies (Python
    builtins, dict methods, path operations).  A ``frozenset`` constant
    helps, but a class hierarchy does not.

``restated_identity``
    Sometimes a false positive from ``TypeVar("ClassName", ...)``, which
    Python requires to spell the class name as a string.

``constant_subtype_answer``
    Boolean properties that return a constant per family member.  Fixing
    requires converting each consumer branch into a polymorphic method.

``zero_one_tuple``
    Methods returning ``()`` or ``(x,)`` as an optional.  Fixing requires
    changing the return contract from iteration to ``T | None``.


Foreign Families Are the Batch Lever
------------------------------------

``ForeignFamily`` (in ``foreign_family.py``) and ``ForeignAdapter`` (in
``declared_family.py``) are the abstractions that make batch conversion
possible.  Each one registers AST classes and provides:

- ``isinstance(node, Family)`` — the one membership test
- ``Family.within(nodes)`` — lazy iteration keeping only members
- ``Adapter.decoded(node)`` — decode one node into the adapter or ``None``
- ``Adapter.within(nodes)`` — decode and iterate

Existing families and their registered members:

``FunctionDefinition``
    ``ast.FunctionDef``, ``ast.AsyncFunctionDef``.  Fields: ``name``,
    ``args``, ``body``, ``decorator_list``, ``lineno``, ``col_offset``.

``AssignmentStatement`` (adapter)
    ``ast.Assign``, ``ast.AnnAssign``.  Properties: ``.targets``,
    ``.value``, ``.annotation``, ``.single_name``.

``ExpressionReference`` (adapter)
    Multiple AST expression types.  Properties: ``.identifier``,
    ``.qualified_name``, ``.terminal_name``, ``.attribute_chain``.

To find candidates for a new foreign family, count how many sites spell the
same isinstance tuple::

    grep -rn 'isinstance.*ast\.Import.*ast\.ImportFrom' src/ | wc -l

If the count is 10+, the family pays for itself.


Batch Scripting Pitfalls
------------------------

When writing a script that converts constructions or isinstance checks
across the codebase:

- **Qualify class names by module.**  Bare class names collide: the
  2026-11-15 keyword conversion had 3 bugs from ``AnnotatedParameter``,
  ``Site``, and ``ModuleDeclarations`` each appearing in two modules with
  different fields.  Use ``inspect.signature(cls)`` on the class imported
  from the specific module.

- **Check for stale variable references after renaming.**  Converting
  ``for statement in ...`` to ``for assignment in ...`` leaves
  ``statement.lineno`` intact if the search-and-replace misses attribute
  accesses.

- **Run the repo guard AND a targeted NRA scan after every batch.**
  The guard (``tests -k test_repository_``) catches import and
  construction errors.  The NRA scan catches runtime paths the guard
  doesn't exercise (e.g., specific detector code).

- **Keyword-first.**  Converting positional constructions to keyword
  (``--json`` with ``inspect.signature``) makes every subsequent MI
  extraction safe, because adding a field to a base class no longer
  changes positional argument order.


Structural Blockers
-------------------

Some refactors are blocked by codebase-wide patterns.  Fix the blocker
first; the extractions it enables are the compound payoff.

Positional construction → keyword
    1043 sites converted in one batch.  Unblocks every MI extraction that
    adds fields to a base class.

``NamedTuple`` bases
    Cannot participate in MI.  Convert to ``@dataclass(frozen=True)``
    first if the tuple needs to inherit shared fields.

Field-name collisions
    Two classes with a field named ``target`` meaning different things
    cannot inherit from each other.  Rename one field before the MI
    extraction.
