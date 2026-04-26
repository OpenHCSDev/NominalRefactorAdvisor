Git Worktree Testing
====================

Use Git worktrees when you need to test a feature branch without disturbing the
main checkout.

Basic Workflow
--------------

Create a worktree for a branch:

.. code-block:: bash

   git worktree add ../advisor-feature feature/my-branch

Verify the worktree list:

.. code-block:: bash

   git worktree list

Run tests inside the worktree:

.. code-block:: bash

   cd ../advisor-feature
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   python -m pytest tests/

Shared-Venv Caution
-------------------

You can share one virtual environment across worktrees, but editable installs
will point at whichever checkout you installed from last. Separate environments
are usually simpler and less error-prone.

Comparing Branches
------------------

To compare output between ``main`` and a feature worktree:

.. code-block:: bash

   cd /path/to/nominal-refactor-advisor
   source .venv/bin/activate
   python -m pytest tests/ -q > /tmp/main-tests.txt 2>&1

   cd ../advisor-feature
   source .venv/bin/activate
   python -m pytest tests/ -q > /tmp/feature-tests.txt 2>&1

   diff /tmp/main-tests.txt /tmp/feature-tests.txt

For docs comparisons:

.. code-block:: bash

   python -m sphinx -E -b html docs/source docs/_build/html

Cleanup
-------

Remove the worktree when finished:

.. code-block:: bash

   git worktree remove ../advisor-feature

If the directory was already deleted manually:

.. code-block:: bash

   git worktree prune

Practical Notes
---------------

- keep each worktree clean before running comparisons
- prefer separate virtual environments per worktree
- compare tests and docs builds, not just diffs
- use worktrees for risky refactors, not only for feature development
