# Anti-pattern → owned structure: a practical refactoring cookbook

**Use the examples to propose a fix, not just to point at a smell.** For each lead,
the agent finds the matching move below, traces the current declarations and
consumers, and proposes a migration based on the task's rules and
specs. It settles routine source questions itself. If a domain decision is truly missing or contested, it raises that before prescribing
the migration with NRA's existing DSL. Start with the good destination, then look at where it
fails before you apply it. This is a growing catalog of the **selected** epoch PRs.
It doesn't claim to cover every change, and it's not a second NRA detector/operation registry.

## Pattern index

| Anti-pattern / search lead | Correctly factored destination when the relation supports it | Evidence / example |
|---|---|---|
| Enum/class members re-listed in validators, exports or lookup tables | Derive the collection from the member declarations. Declare a separate subset policy only if it means something on its own. | #60 names/aliases; example 1 |
| Source×target switch with duplicated orchestration | One algorithm over a small set of nominal implementation hooks | #38 conversion ABC; example 2 |
| Hand-maintained family registry beside nominal classes | Class-definition registration with a declared key and an explicit collision policy | #60 module registration; example 3 |
| Repeated `hasattr`/concrete-type interpretation by callers | Public capability ABC. The consumer calls its contract and the concrete leaf supplies the operation. | #44 get/set capability (not its retained reset switch); example 2 |
| Overlapping capability implementations copied into every combination | Shared implementations composed through C3/MI on the same nominal object | #60 image/object inputs; example 4 |
| String handler plus unconstrained `options` dict | Typed option families, shared inherited fields, explicit type→writer binding | #69; example 5 |
| UI manager stores and synchronizes domain state | An independently lived model owns state/transitions. The UI projects and invokes it. | #58; example 6 |
| Per-case classes repeat setup/validation around one varying operation | Parent template method owns the common algorithm. Abstract hooks cover only what really varies. | #60 `ModuleSettingsSourceModule`; example 2 |
| Boolean combinations hide named commands or capabilities | Explicit commands or cooperative capability mixins, where the domain defines the combinations/order | User example; mechanism guide (not claimed as an exact PR flag rewrite) |
| Expected outcomes hidden behind catch routing; raw string statuses | Named outcomes/states where callers need that distinction | User example; mechanism guide (not claimed as an exact PR state rewrite) |
| `FooAuthority`/`FooDeclaration` wrappers just forward data | Reuse the real domain owner and remove the redundant forwarding | User naming criterion; judge behavior, not the suffix |

Read [nominal mechanisms](nominal-mechanisms.md) for the last three shapes and
for Enum/state, validation, MI and metaclass semantics. The snippets below clearly
separate runnable **teaching examples** from abbreviated **historical source**.
Teaching examples specify a tiny new domain. They're not production patches, and they don't
prove that any arbitrary before-source is equivalent.

## 1. A declared set should determine its projections

**Bad teaching fragment:** every extra member means another edit.

```python
class Format(Enum):
    CSV = "csv"
    JSON = "json"

VALID_FORMATS = {"csv", "json"}  # Independently editable copy of all members.
```

**Better — runnable teaching example:** keep the vocabulary in one declaration.

```python
from enum import Enum

class Format(Enum):
    CSV = "csv"
    JSON = "json"
    TEXT = "text"  # Added only here; projections below include it.

def accepted_formats():
    return frozenset(member.value for member in Format)

def parse_format(raw):
    return Format(raw)  # Boundary conversion; unknown value raises ValueError.

assert accepted_formats() == frozenset({"csv", "json", "text"})
assert parse_format("text") is Format.TEXT
```

**Maintenance gain:** you no longer have to remember a second roster for
validation/display membership. A hardcoded expected set in a test is an assertion, not a second
production authority. If a UI deliberately allows only CSV/JSON, that's a
separate selection policy. It isn't automatically a stale copy of the full vocabulary.

**Historical basis:** #60 `5a57218b3→63f0ede15`,
`openhcs/interop/cellprofiler/module_semantics.py`: independently authored family
and alias tuples gave way to projections over module declarations. The old
`MappingProxyType` maps were already derived, so don't call them independent
mirrors. **NRA trajectory:** pick the declaration and the redundant roster, derive
its consumer projection using supported operations, migrate consumers, remove the
old writable roster. Test new members, aliases and unknown values.

## 2. Public ABC + shared algorithm + irreducible hooks

**Bad teaching fragment:** the common precondition and sequence are copied into
several case branches (or into subclasses verbatim).

```python
if kind == "upper":
    if not isinstance(value, str):
        raise TypeError("expected text")
    return value.upper()
elif kind == "lower":
    if not isinstance(value, str):
        raise TypeError("expected text")
    return value.lower()
```

**Better — runnable teaching example:** the consumer knows the public operation.
The parent owns the invariant and the leaf supplies only the variation.

```python
from abc import ABC, abstractmethod

class TextTransform(ABC):
    def apply(self, value):
        if not isinstance(value, str):
            raise TypeError("expected text")
        return self.transform(value)

    @abstractmethod
    def transform(self, value):
        raise NotImplementedError

class Upper(TextTransform):
    def transform(self, value):
        return value.upper()

class Lower(TextTransform):
    def transform(self, value):
        return value.lower()

assert Upper().apply("MiXeD") == "MIXED"
assert Lower().apply("MiXeD") == "mixed"
```

**Maintenance gain:** adding a transform means writing one hook. You don't copy another
validation sequence or add a consumer-side branch. You still need a defined boundary for construction/selection. This fragment
deliberately doesn't solve a wire string→instance registry or unknown kind handling.

**Real positive source:** #60 child `63f0ede15`,
`openhcs/processing/backends/cellprofiler/module_classes.py`,
`ModuleSettingsSourceModule.settings_source` is abstract while `bind_settings`
calls it and owns the shared finalization. #38 child `05eb87f07^2`,
`core/memory/conversion_helpers.py`, `MemoryTypeConverter` supplies four abstract
hooks used by the generated conversion orchestration. That after-source still has an
`_OPS` table and dynamic `eval`. Take the algorithm/hook split from it, not the
idea that every table or effect went away. **NRA trajectory:** move shared
members to the parent, migrate exact calls, keep creation/binding
and every unknown/default path, then check descendants and before/after effects.

## 3. A metaclass can derive registration from a class declaration

**Bad shape:** declare module classes, then separately hand-author all their
names and aliases in another roster, with runtime leaf policy names somewhere else.
Every new class now needs synchronized updates.

**Historical source, abridged/non-runnable:** #60 child `63f0ede15`,
`openhcs/processing/backends/cellprofiler/module_classes.py`:

```python
class CellProfilerModule(ABC, metaclass=AutoRegisterMeta):
    __registry__ = LazyDiscoveryDict(enable_cache=False)
    __registry_key__ = "module_name"
    __skip_if_no_key__ = True
    module_name: ClassVar[str | None] = None
    aliases: ClassVar[tuple[str, ...]] = ()
    # __init_subclass__: normalize/validate declarations; check name collisions.
```

Its semantics consumer now derives alias pairs:

```python
# Abridged from module_semantics.py::_declared_aliases
return tuple(
    (alias, str(module_type.module_name))
    for module_type in CellProfilerModule.__registry__.values()
    for alias in module_type.aliases
)
```

For the matching settings-owner migration, the independent roles that stay and the
real semantic differences, see the
[worked #60 trajectory](theory-to-workflow.md#worked-60-settings-and-capability-migration).

**Maintenance gain:** the declared class carries the name/aliases that consumers
project. Shared validation lives at the class-definition boundary, not in every
consumer. Reuse the project's registration mechanism. Don't write another
metaclass. **NRA trajectory:** migrate the facts the declaration owns first, derive
the registry consumers, then remove the old name/alias rosters only after finding all
their uses. Check duplicate/inherited keys, imported/dynamic classes, aliases and unknown
names. Setup-only `SetupModuleCompiler` stays a separate family. #44's real
`WidgetMeta` warns and then overwrites duplicate IDs, so automatic registration
alone doesn't guarantee uniqueness.

## 4. Multiple inheritance composes real implementation capabilities

**Bad shape:** image-only, object-only and combined consumers each copy
the same image/object input computation, or branch over the combinations.

**Runnable teaching example of the C3 mechanism:** these fixed input labels
stand in for a tiny declared contract. Production #60 derives artifacts from settings.

```python
class Inputs:
    @classmethod
    def inputs(cls):
        return ()

class ImageInputs(Inputs):
    @classmethod
    def inputs(cls):
        return ("image", *super().inputs())

class ObjectInputs(Inputs):
    @classmethod
    def inputs(cls):
        return (*super().inputs(), "objects")

class CombinedInputs(ImageInputs, ObjectInputs):
    pass

assert ImageInputs.inputs() == ("image",)
assert ObjectInputs.inputs() == ("objects",)
assert CombinedInputs.inputs() == ("image", "objects")
assert CombinedInputs.__mro__ == (
    CombinedInputs, ImageInputs, ObjectInputs, Inputs, object
)
```

**Real counterpart:** #60 child `63f0ede15` composes
`ScopedMeasurementModule(ImageMeasurementInputModule, ObjectMeasurementInputModule)`.
Both parents implement `measurement_artifact_inputs` cooperatively. Image inputs
come before `super()` output and object inputs come after it. The base owns the shared
endpoint. **Maintenance gain:** one image-input implementation takes part in
several legitimate combinations instead of being copied into each leaf. **NRA
trajectory:** set up the bases/order, promote common implementations
and remove duplicates, prove the member lookup you selected, and test output/effect order.
This example shows one MRO. It doesn't show that base orders are interchangeable or that
cooperative constructors work in general.

## 5. Typed configuration can compose shared fields without copying bags

**Before #69, abridged source** (`b5abe6aa6^1`, materialization `core.py`):

```python
@dataclass(frozen=True)
class MaterializationSpec:
    handler: str
    options: Dict[str, Any] = field(default_factory=dict)
# materialize: MaterializationRegistry.get(spec.handler), then handler.func(...)
```

**After, abridged source** (`b5abe6aa6^2`, `options.py`):

```python
@dataclass(frozen=True)
class CsvOptions(FileOutputOptions, SourceOptions, TabularExtractionOptions):
    filename_suffix: str = "_details.csv"

@dataclass(frozen=True)
class JsonOptions(FileOutputOptions, SourceOptions, TabularExtractionOptions):
    filename_suffix: str = ".json"
    indent: int = 2
    wrap_list: bool = False
```

**Maintenance gain:** shared file/source/tabular fields are declared once. A
format picks the capabilities it needs and adds only its own options.
`MaterializationSpec` explicitly rejects dicts and unregistered option types.
`materialize` selects `_WRITERS_BY_OPTIONS[type(opt)]`. Writer registration and
`MaterializationFormat` are still explicit. Nothing infers them from class
names. Dataclass annotations don't validate field values on their own. **NRA
trajectory:** introduce typed carriers and shared bases, migrate constructors and
exact callers, adapt the boundary, then remove the old string/dict paths. Test new
format registration, default field ordering, malformed options and serialization.

## 6. Separate state lifetime from the view; derive instead of synchronizing

**Before shape:** the view owns mutable parameters/defaults/reset flags and
has to sync other views or saved snapshots whenever it changes them.

**After #58, abridged source** (`43a6cba5b`, PFM):

```python
@property
def parameters(self):
    return self.state.parameters
```

That property is a direct projection. `ObjectState` owns parameters,
reset/user-set fields and mutation methods like `update_parameter` and
`reset_parameter`. PFM takes `state` in its constructor. **Maintenance gain:**
reopening the view doesn't have to recreate the model's identity, and the view reads the state
instead of keeping another writable copy. **NRA trajectory:** set up the
state lifetime/construction boundary, keep the state object, migrate reads and
mutations, then delete the redundant storage. Trace each real caller. Nested state
and separate scope-registry lookup paths aren't proved identical by `self.state`.
The sixth dynamic `getattr` property and saved/live snapshots remain separate
obligations. A snapshot can be deliberately independent. Don't delete it as a mirror.

## Transfer the move, not the incidental spelling

For each application show: **old independent edit sites → new determining
operation/declaration → derived consumers → one new-case experiment**. Keep
NRA's [canonical pattern catalog](../../../docs/source/api/pattern_catalog.rst)
and registered DSL operations as the tool authorities. This cookbook trains
the agent's judgment. It doesn't claim every example has an automatic
recipe. See [ownership-to-DSL](ownership-to-dsl.md) for the simulation/apply gates.
