Codemod Operation Catalogue
===========================

This reference is generated from the registered ``RefactorRecipeOperation``
declarations. Each entry exposes its Python constructor, operation key,
implementation path, inheritance and source proof scope.

The operator supplies the semantic decision and operation parameters. Preflight
checks the required source relations. Sequence simulation resolves each stage
against the preceding stage's projected source; projected findings can then
supply an executable continuation.

See :doc:`getting_started` for running plans and :doc:`public_api` for the shared
proof and execution contracts.

.. contents:: Operations
   :local:
   :depth: 1

.. include:: _generated/codemod_catalog.rst
