# Theory → evidence → migration

**NRA shows you the evidence. It doesn't know what your business means.** The task owner supplies the intended rules and points at the authoritative declarations or specs. The agent investigates the source, applies the theory, and proposes a concrete factoring. NRA's deterministic checks cover supported source observations, provenance, uncertainty and transformations against the inputs you accepted. Nobody's job replaces anybody else's.

That doesn't mean making the practitioner list classes or sign off on every edge. The agent works out what follows from the rules the task already gave, chases counterexamples and supplies a design. Ask only when a domain decision is genuinely missing or contested and it changes which architecture is correct.

## The shared Paper 1 anchor

Tristan Simas, *Semantic Completeness for Correct Maintenance: A Strict Hierarchy
of Typing and Inheritance*, §2, Theorem 4.8 and Theorem 6.2, splits this into three
questions:

1. **What answers are required?** Fix the meanings, the changes that can happen and the questions that must be answerable. Correct recovery is `r(e(x)) = a(x)`. If the representation merges two cases that need different answers, another helper on top of it can't get the lost distinction back.
2. **Which connections does the mechanism derive?** Keep the required relation apart from observed ancestry and from explicit forwarding/registration. The agent has to preserve both required and forbidden connections. Missing source evidence is not a forbidden connection.
3. **Does the implementation behave correctly?** Covering every connection doesn't prove substitution, effect order, constructors or protocol parity. And passing the same tests doesn't prove the cases share one future.

Theorem 4.8 gives the design rule you can actually use: merge meanings only when their complete accepted answer histories agree. Share implementation but keep the identities when only the implementation agrees. Keep different implementation answers separate. An authored DSL batch cuts mechanical work. It doesn't cut the number of answers you have to justify (Propositions 4.9 and 4.17).

### A concrete reason for meaningful multiple inheritance

Paper Figure 1 fixes this small teaching contract:

| Independent implementation role | Execute | Checkpoint | Export |
|---|---|---|---|
| Selection configuration | required | required | forbidden |
| Destination configuration | forbidden | required | required |

`Execute(SelectionConfig)`, `Checkpoint(SelectionConfig, DestinationConfig)` and
`Export(DestinationConfig)` directly express the four required connections.
Neither provider's consumer group contains the other. For this fixed relation, a sound one-parent ancestry leaves at least one required connection outside ancestry. Forwarding can supply it, but the obligation is still there. That's why MI can remove real coordination instead of just shortening code.

The table is a teaching spec someone handed us, **not something you can infer from three class names**. Real Python composition still needs valid C3 lookup, cooperative initialization, and sane state and member-conflict behavior. Don't compute an ancestry-gap claim for a guessed or incomplete production relation.

## Worked #60 settings and capability migration

**Historical source, not a replay recipe.** Selected direct OpenHCS transition:
`5a57218b3ead69f2ae97aee0a214f4bdedae0b2a` →
`63f0ede1514abd7b4f4209c71eac05468d30f3e8`. Paths below are relative to that
repository's `openhcs/` package directory. Re-read these Git objects when you transfer the lesson. Nobody has proved the runtime or the whole PR equivalent here.

### Before: related answers lived in separate editable places

- `interop/cellprofiler/module_semantics.py:174–428` authored family metadata
  and alias tuples. The maps at `431–487` were **already derived**, not extra mirrors.
- `interop/cellprofiler/module_settings_binding.py:603–655` selected a separate
  `_ModuleSettingsBindingStrategy` registry with a generic fallback.
  `MeasureObjectSizeShapeModuleSettingsBindingStrategy` (`1779–1815`) and
  `CropModuleSettingsBindingStrategy` (`2722–2734`) supplied special settings.
- `interop/cellprofiler/runtime/module_execution.py:2860–2906` independently specified object-label
  policy selections, including Crop's `cropping_labels` versus measurements'
  `labels`. That difference isn't redundant just because both accept labels.

### After: declarations own variation, parents own algorithms

- `processing/backends/cellprofiler/module_classes.py:239–360` defines the public
  `CellProfilerModule` family, local declaration validation and registry lookup.
- Its shared binder/finalization methods (`379–612`) own the common settings work.
  `ModuleSettingsSourceModule` (`1103–1137`) declares abstract `settings_source`.
  Its concrete `bind_settings` calls that hook and shared finalization.
  `BinderSettingsSourceModule` is a separate template for hooks that need a binder.
- `processing/backends/cellprofiler/crop.py:95–192` puts Crop identity,
  selected capabilities and its settings hook in one place. Its template is
  **`BinderSettingsSourceModule.bind_settings` → `CropModule.settings_source`
  with the binder argument**, not the binder-free template above.
  `shape.py:89–132` instead supplies `setting_bindings` and `ignored_settings`
  consumed by the shared `CellProfilerModule.bind_settings` machinery. It
  uses neither `settings_source` template. Different variants can
  derive from different shared algorithms without another per-module dispatcher.
- `interop/cellprofiler/pipeline_generator.py:314–365` picks the required module
  class and calls `.bind_settings(...)` and `.resolve_function(...)`, instead of
  asking another settings-strategy family to reinterpret the module name.
- `interop/cellprofiler/runtime/object_input_policies.py:146–164,198–207` shares the single-object
  binding algorithm while keeping distinct named keyword policies. Crop uses
  `CroppingObjectLabelInputPolicy`. Shape uses `LabelsObjectInputPolicy`.

**The good factoring:** module declarations determine the related case choices. Shared settings/binding algorithms are inherited. Generic callers invoke the public contract. The existing metaclass derives lookup. It isn't where the ownership decision came from. The old strategy family already used registration, so bolting on a metaclass wouldn't have removed the split.

### Independent capabilities and boundaries stay visible

`ScopedMeasurementModule(ImageMeasurementInputModule, ObjectMeasurementInputModule)`
is declared at `module_classes.py:1779`. Its two contributing implementations
are at `1719–1775`, with the base endpoint at `867–881`. Image inputs come before
`super()` output. Object inputs come after it. For that chain, the predicted order is **image + base-declared inputs + object inputs**.
That's a read of the source chain. Nobody ran it and proved the complete concrete MRO.

`interop/cellprofiler/module_semantics.py:163–195` derives rows from backend
registration **and a separate `SetupModuleCompiler` registry**. Setup compilation
isn't automatically another backend subtype. Shared lookup output doesn't make
independent provider roles the same thing.

### The domain rule decides what counts as a regression

The old Crop row (`module_semantics.py:233–248`) declares `respects_masks=True`.
The child's backend semantic projection writes `False` (`:163–176`). The source plainly shows a different answer. Two possible task contracts give different
conclusions:

- If this query has to keep the old mask-support answer, `False` is a
  divergence. Fix or test it before you call the migration equivalent.
- If the task deliberately introduces a different, narrower query, keep its
  own identity and migrate consumers explicitly. Don't call it unchanged
  just because the field spelling survived.

Nothing here establishes which one the history intended. The source
diff points at the decision. The task's contract settles it. Unknown-name
fallbacks, dimensionality, aliases and family grouping also need explicit checks.

### Express the complete migration, not just the class insertion

For a similar new target, the agent delivers a filled-in decision receipt:

`rule/authority from the task → original declarations and consumers → required and
forbidden pairs → proposed owner and projections → remaining OPEN obligations`.

Keep the relation's two semantic columns `(implementation_key, consumer)`.
Source positions and rule/acceptance references are supporting metadata.

Then compose the target's existing NRA operations:

1. Reuse or set up the public declaration family and required imports.
2. Move common members into the parent. Keep the case hooks that can't be shared.
3. Set up capability bases and validate exact member lookup/constructor order.
4. Migrate settings callers and derive the name/alias/semantic projections.
5. Delete old strategies/rosters only after checking every remaining use and
   initializer effect. Keep distinct setup, wire and runtime-policy owners.
6. Simulate the dependent stages, inspect the combined diff, and run preservation
   tests plus a new-module experiment. Report every edit site that's still hand-authored.

This is a migration shape, not an automatic #60 recipe. A new module's
name, aliases and setting facts should reach the intended consumers from its declared
owner. Independent policies may still legitimately need their own declarations.

## Translate source leads into the existing NRA machinery

| Concrete lead | Authority to reuse | Narrow mechanical route / boundary |
|---|---|---|
| Enum values repeated in a collection | Original Enum/member and roster syntax, canonical class/lexical evidence | `DeriveEnumSubsetOperation` handles its proved literal-string `frozenset` subset shape. It doesn't handle arbitrary mutating sets or foreign values. Decide first whether you have the full vocabulary or an independent subset. |
| Manual registry beside an ABC family | Class-family index, original registry writes and import/creation order | `ConvertManualRegistryToAutoregisterOperation` needs its supported source component and native-use checks. It's not a generic enum-to-ABC conversion. |
| Collection of family classes/names | Canonical class family and original collection, keeping order where it's ordered | `DeriveClassFamilyCollectionOperation` doesn't cover arbitrary attribute-value keys, dynamic plugin discovery or incomplete subsets. |
| Repeated action branches | Original ordered branch/Call source plus lexical/product-flow owners | `DispatchToPolymorphismOperation` supports synchronous module functions and plain undecorated instance methods of top-level classes (no `super`/`__class__`, no mangled names), never async functions. The axis is a parameter or a parameter-rooted attribute chain. Leading binding-free terminal guards stay in place. Each case body must be terminal on every path and read no local bound outside itself. Being eligible doesn't prove effect parity. Use the separately labeled [authored example](action-batch.md) to learn batching, not to stretch eligibility. |

Check the selected checkout's registered operation, constructor and preflight
before you use any of these. This table is a set of worked entry points, not another operation or
binding registry. Keep extra/missing roster items, unsupported original classes,
ordered fallbacks, duplicate keys and dynamic writes as evidence. Don't clean them out of the candidate relation.

Finish with **old independent edit sites → determining declaration/algorithm →
derived consumers → new-case experiment**. Keep the task's rules, source/binding
checks, architecture and behavioral evidence separate. See
[batching](batching.md), [ownership-to-DSL](ownership-to-dsl.md), and the maintained
[semantic derivation protocol](../../../docs/source/development/semantic_derivation_protocol.rst).
