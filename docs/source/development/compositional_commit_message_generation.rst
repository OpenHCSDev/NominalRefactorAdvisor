Compositional Git Commit Message Generation
===========================================

Systematic methodology for generating comprehensive, technically accurate git commit messages using semantic file grouping and compositional reasoning.

Core Methodology
----------------

Semantic File Grouping
~~~~~~~~~~~~~~~~~~~~~~

Group modified files by **functional purpose** within the system:

- **Memory Management**: Memory type conversion, stack utilities, memory wrappers
- **Processor Backends**: Backend-specific implementations (cupy, numpy, torch, etc.)
- **Core Pipeline**: Orchestration, step execution, function calling logic
- **Logging & Debugging**: Logging infrastructure, debug utilities, monitoring
- **API Interfaces**: Public APIs, function signatures, contracts
- **Documentation**: README files, API docs, architecture docs

Change Analysis Per Group
~~~~~~~~~~~~~~~~~~~~~~~~~

For each semantic group:

1. **Read the actual code changes** - Examine the diffs, don't assume
2. **Understand the technical impact** - How do these changes affect system behavior?
3. **Identify the root cause** - What problem was being solved?
4. **Assess scope of impact** - What other parts of the system are affected?

Component Message Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create focused commit message components for each group:

- **Start with the functional area**: "Memory Management:", "CuPy Processor:", etc.
- **Describe the change type**: "Fix signature mismatch", "Add logging", "Refactor interface"
- **Explain the technical details**: What specifically was changed and why
- **Note the impact**: How this affects system behavior or fixes issues

Message Synthesis
~~~~~~~~~~~~~~~~~

Combine all components into a structured commit message:

.. code-block:: text

   <Primary Change Type>: <High-level summary>

   <Detailed description of the main change and its motivation>

   Changes by functional area:

   * <Functional Area 1>: <Component message 1>
   * <Functional Area 2>: <Component message 2>
   * <Functional Area 3>: <Component message 3>

   <Additional context, breaking changes, or follow-up notes if needed>

Example Application
-------------------

Step 1: Identify Modified Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   nominal_refactor_advisor/detectors/_base.py
   nominal_refactor_advisor/detectors/_helpers.py
   nominal_refactor_advisor/detectors/_runtime.py
   nominal_refactor_advisor/patterns.py
   nominal_refactor_advisor/planner.py
   tests/test_refactor_advisor.py

Step 2: Semantic Grouping
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Detector Substrate**: _base.py, _helpers.py, _runtime.py
- **Pattern/Planner Surface**: patterns.py, planner.py
- **Regression Coverage**: test_refactor_advisor.py

Step 3: Analyze Changes
~~~~~~~~~~~~~~~~~~~~~~~

- **Detector Substrate**: Collapsed shared helper logic and widened generic detector support
- **Pattern/Planner Surface**: Updated authoritative metadata or planner composition behavior
- **Regression Coverage**: Added tests that lock in the new detection or refactor surface

Step 4: Generate Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Detector Substrate**: Refactor shared detector machinery and keep generic detection coherent
- **Pattern/Planner Surface**: Update canonical pattern or planning authority
- **Regression Coverage**: Add regression tests for the new structural behavior

Step 5: Synthesize Final Message
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   refactor: unify detector substrate and planner authority

   Changes by functional area:

   * Detector Substrate: collapse repeated helper logic and keep detector registration
     and genericity machinery in one authoritative layer
   * Pattern/Planner Surface: align plan composition with the canonical pattern metadata
   * Regression Coverage: add tests that lock in the new structural contracts

Benefits
--------

1. **Comprehensive Coverage**: Every change is documented and contextualized
2. **Technical Accuracy**: Messages reflect actual code changes, not assumptions
3. **Logical Organization**: Related changes are grouped for better understanding
4. **Debugging Aid**: Future developers can understand the scope and reasoning
5. **Systems Thinking**: Changes are understood in context of overall architecture

Usage
-----

This methodology should be used for:

- Complex changes spanning multiple files
- Bug fixes that require changes across different system layers
- Refactoring that affects multiple components
- Any change where the scope and impact need to be clearly communicated

For simple, single-file changes, a standard commit message format is sufficient.
