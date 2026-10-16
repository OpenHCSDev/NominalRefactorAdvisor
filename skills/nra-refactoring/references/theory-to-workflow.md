# Theory → evidence → migration

**NRA is an evidence-preserving lens, not an oracle for business meaning.** The
task owner supplies intended rules and identifies authoritative declarations or
specifications. The agent performs the source investigation, applies the theory,
and proposes concrete factoring. NRA's deterministic authorities check supported
source observations, provenance, uncertainty and transformations against admitted
inputs. None of these roles replaces the others.

This does not mean making the practitioner inventory classes or approve every
edge. The agent derives consequences of already authorized rules, investigates
counterexamples and supplies a design. Ask only when a genuinely missing or
contested domain decision changes which architecture is correct.

## The shared Paper 1 anchor

Tristan Simas, *Semantic Completeness for Correct Maintenance: A Strict Hierarchy
of Typing and Inheritance*, §2, Theorem 4.8 and Theorem 6.2, separates three
questions:

1. **What answers are required?** Fix the meanings, reachable changes and required
   questions. Correct recovery is `r(e(x)) = a(x)`; if the representation merges
   two cases needing different answers, another helper over that representation
   cannot recover the lost distinction.
2. **Which connections does the mechanism derive?** Keep the required relation
   distinct from observed ancestry and explicit forwarding/registration. The
   agent must preserve both required and forbidden connections; missing source
   evidence is not a forbidden connection.
3. **Does the implementation behave correctly?** Complete connection coverage
   does not prove substitution, effect order, constructors or protocol parity.
   Conversely, passing the same tests does not establish one shared future.

Theorem 4.8 gives the useful design rule: merge meanings only when their complete
admitted answer histories agree; share implementation while retaining identities
when only implementation agrees; keep different implementation answers separate.
An authored DSL batch reduces mechanical work, not the number of semantic answers
that must be justified (Propositions 4.9 and 4.17).

### A concrete reason for meaningful multiple inheritance

Paper Figure 1 fixes this small teaching contract:

| Independent implementation role | Execute | Checkpoint | Export |
|---|---|---|---|
| Selection configuration | required | required | forbidden |
| Destination configuration | forbidden | required | required |

`Execute(SelectionConfig)`, `Checkpoint(SelectionConfig, DestinationConfig)` and
`Export(DestinationConfig)` directly express the four required connections.
Neither provider's consumer group contains the other. For this fixed relation,
a sound one-parent ancestry leaves at least one required connection outside
ancestry; forwarding can supply it but does not erase the obligation. This is
why MI can remove real coordination, rather than merely shorten code.

The table is a supplied teaching specification, **not something inferred from
three class names**. Real Python composition still requires valid C3 lookup,
cooperative initialization, state and member-conflict behavior. Do not compute
an ancestry-gap claim for a guessed or incomplete production relation.

## Worked #60 settings and capability migration

**Historical source, not a replay recipe.** Selected direct OpenHCS transition:
`5a57218b3ead69f2ae97aee0a214f4bdedae0b2a` →
`63f0ede1514abd7b4f4209c71eac05468d30f3e8`. Paths below are relative to that
repository's `openhcs/` package directory. Reinspect these Git objects when transferring the lesson; the
runtime and the entire PR have not been proved equivalent here.

### Before: related answers lived in separate editable places

- `interop/cellprofiler/module_semantics.py:174–428` authored family metadata
  and alias tuples. Maps at `431–487` were **already derived**, not extra mirrors.
- `interop/cellprofiler/module_settings_binding.py:603–655` selected a separate
  `_ModuleSettingsBindingStrategy` registry with a generic fallback.
  `MeasureObjectSizeShapeModuleSettingsBindingStrategy` (`1779–1815`) and
  `CropModuleSettingsBindingStrategy` (`2722–2734`) supplied special settings.
- `interop/cellprofiler/runtime/module_execution.py:2860–2906` independently specified object-label
  policy selections, including Crop's `cropping_labels` versus measurements'
  `labels`. That difference is not redundant just because both accept labels.

### After: declarations own variation, parents own algorithms

- `processing/backends/cellprofiler/module_classes.py:239–360` defines the public
  `CellProfilerModule` family, local declaration validation and registry lookup.
- Its shared binder/finalization methods (`379–612`) own common settings work.
  `ModuleSettingsSourceModule` (`1103–1137`) declares abstract `settings_source`;
  its concrete `bind_settings` calls that hook and shared finalization.
  `BinderSettingsSourceModule` is a separate template for hooks needing a binder.
- `processing/backends/cellprofiler/crop.py:95–192` co-locates Crop identity,
  selected capabilities and its settings hook. Its selected template is
  **`BinderSettingsSourceModule.bind_settings` → `CropModule.settings_source`
  with the binder argument**, not the binder-free template above.
  `shape.py:89–132` instead supplies `setting_bindings` and `ignored_settings`
  consumed by the shared `CellProfilerModule.bind_settings` machinery; it does
  not use either `settings_source` template. Different admitted variants may
  derive from different shared algorithms without another per-module dispatcher.
- `interop/cellprofiler/pipeline_generator.py:314–365` selects the required module
  class and calls `.bind_settings(...)` and `.resolve_function(...)`, rather than
  asking another settings-strategy family to reinterpret the module name.
- `interop/cellprofiler/runtime/object_input_policies.py:146–164,198–207` shares the single-object
  binding algorithm while keeping distinct named keyword policies. Crop uses
  `CroppingObjectLabelInputPolicy`; Shape uses `LabelsObjectInputPolicy`.

**The positive factoring:** module declarations determine related case choices;
shared settings/binding algorithms are inherited; generic callers invoke the
public contract. The existing metaclass is a mechanism for deriving lookup—not
the source of the ownership decision. The old strategy family already used
registration, so adding a metaclass alone would not have removed the split.

### Independent capabilities and boundaries stay visible

`ScopedMeasurementModule(ImageMeasurementInputModule, ObjectMeasurementInputModule)`
is declared at `module_classes.py:1779`; its two contributing implementations
are at `1719–1775`, with the base endpoint at `867–881`. Image inputs precede
`super()` output; object inputs follow it. For that declared
chain, the predicted order is **image + base-declared inputs + object inputs**.
This is a source-level chain analysis, not an executed complete-concrete-MRO proof.

`interop/cellprofiler/module_semantics.py:163–195` derives rows from backend
registration **and a separate `SetupModuleCompiler` registry**. Setup compilation
is not automatically another backend subtype. Shared lookup output does not make
independent provider roles identical.

### The domain rule decides what counts as a regression

The old Crop row (`module_semantics.py:233–248`) declares `respects_masks=True`.
The child's backend semantic projection writes `False` (`:163–176`). That is a
source-visible answer difference. Two possible task contracts give different
conclusions:

- If this query must preserve the old mask-support answer, `False` is a
  divergence to fix/test before calling the migration equivalent.
- If the task deliberately introduces a different, narrower query, retain its
  distinct identity and migrate consumers explicitly; do not call it unchanged
  merely because the field spelling survives.

Neither interpretation is established as the historical intent here. The source
diff identifies the decision; the supplied contract settles it. Unknown-name
fallbacks, dimensionality, aliases and family grouping also need explicit checks.

### Express the complete migration, not just the class insertion

For an analogous new target, the agent delivers a populated decision receipt:

`supplied rule/authority → original declarations and consumers → required and
forbidden pairs → proposed owner and projections → remaining OPEN obligations`.

Keep the relation's two semantic columns `(implementation_key, consumer)`;
source positions and rule/admission references are supporting metadata.

Then compose the target's existing NRA operations:

1. Reuse or establish the public declaration family and required imports.
2. Move common members into the admitted parent; retain irreducible case hooks.
3. Establish capability bases and validate exact member lookup/constructor order.
4. Migrate settings callers and derive the admitted name/alias/semantic projections.
5. Delete old strategies/rosters only after checking every remaining use and
   initializer effect. Preserve distinct setup, wire and runtime-policy owners.
6. Simulate the dependent stages, inspect the combined diff, and run preservation
   tests plus a new-module experiment. Report every still-authored edit site.

This is a migration shape, not a supplied #60 automatic recipe. A new module's
name, aliases and setting facts should reach intended consumers from its declared
owner; independent policies may still legitimately require their own declarations.

## Translate source leads into the existing NRA machinery

| Concrete lead | Authority to reuse | Narrow mechanical route / boundary |
|---|---|---|
| Enum values repeated in a collection | Original Enum/member and roster syntax, canonical class/lexical evidence | `DeriveEnumSubsetOperation` handles its proved literal-string `frozenset` subset shape, not arbitrary mutating sets or foreign values. Determine full vocabulary versus independent subset first. |
| Manual registry beside an ABC family | Class-family index, original registry writes and import/creation order | `ConvertManualRegistryToAutoregisterOperation` requires its supported source component and native-use checks; it is not a generic enum-to-ABC conversion. |
| Collection of family classes/names | Canonical class family and original collection, preserving order where ordered | `DeriveClassFamilyCollectionOperation` is not a proof for arbitrary attribute-value keys, dynamic plugin discovery or incomplete subsets. |
| Repeated action branches | Original ordered branch/Call source plus lexical/product-flow owners | `DispatchToPolymorphismOperation` supports a narrow synchronous module-function shape, not methods or any async function; each case requires a single return. Eligibility alone does not prove effect parity. Use the separately labeled [authored example](action-batch.md) to learn batching, not to widen eligibility. |

Inspect the selected checkout's registered operation, constructor and preflight
before use. This table is a set of worked entry points, not another operation or
binding registry. Retain extra/missing roster items, unsupported original classes,
ordered fallbacks, duplicate keys and dynamic writes as evidence rather than
cleaning them out of the candidate relation.

Finish with **old independent edit sites → determining declaration/algorithm →
derived consumers → new-case experiment**. Keep task admission, source/binding
checks, architecture and behavioral evidence separate. See
[batching](batching.md), [ownership-to-DSL](ownership-to-dsl.md), and the maintained
[semantic derivation protocol](../../../docs/source/development/semantic_derivation_protocol.rst).
