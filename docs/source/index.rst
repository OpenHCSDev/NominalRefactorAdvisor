Nominal Refactor Advisor
========================

The nominal refactor advisor is an AST-driven maintenance tool for detecting and
prescribing theory-grounded nominal refactorings in Python codebases.

It serves mature systems that have accumulated valuable production behaviour
alongside architectural debt.  The operator supplies the semantic decision;
NRA preserves the existing source contract and automates the mechanically
determined edits needed to reach it.  This makes large refactors tractable
without discarding the integrations and operational knowledge already paid for.

If you are new to the project, start with :doc:`api/getting_started`.

The documentation is organized into three layers:

- a guide for running the tool and understanding the stable entrypoints
- reference pages for the shipped API, generated catalogs, and internal substrates
- development notes for architectural rationale, maintenance workflow, and self-hosting policy

.. toctree::
   :maxdepth: 2
   :caption: Guide

   api/getting_started

.. toctree::
   :maxdepth: 2
   :caption: API

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Development

   development/index
