# Anti-pattern → owned structure: a practical refactoring cookbook

**Use examples to propose a solution, not merely to report a smell.** For each lead,
the agent identifies the applicable move below, traces current declarations and
consumers, fills the missing domain decisions itself, and drafts the migration
with NRA's existing DSL. Start with the positive destination; inspect its failure
cases before applying. This is a growing catalog of the **selected** epoch PRs,
not a claim to cover every change or a second NRA detector/operation registry.

## Pattern index

| Anti-pattern / search lead | Correctly factored destination when the relation supports it | Evidence / example |
|---|---|---|
| Enum/class members re-listed in validators, exports or lookup tables | Derive the collection from the member declarations; declare a distinct subset policy only if it has independent meaning | #60 names/aliases; example 1 |
| Source×target switch with duplicated orchestration | One algorithm over a small set of nominal implementation hooks | #38 conversion ABC; example 2 |
| Hand-maintained family registry beside nominal classes | Class-definition registration with a declared key and explicit collision policy | #60 module registration; example 3 |
| Repeated `hasattr`/concrete-type interpretation by callers | Public capability ABC; consumer calls its contract; concrete leaf supplies the operation | #44 get/set capability (not its retained reset switch); example 2 |
| Overlapping capability implementations copied into every combination | Shared implementations composed through C3/MI on the same nominal object | #60 image/object inputs; example 4 |
| String handler plus unconstrained `options` dict | Typed option families, shared inherited fields, explicit type→writer binding | #69; example 5 |
| UI manager stores and synchronizes domain state | Independently lived model owns state/transitions; UI projects and invokes it | #58; example 6 |
| Per-case classes repeat setup/validation around one varying operation | Parent template method owns common algorithm; abstract hooks express only real variation | #60 `ModuleSettingsSourceModule`; example 2 |
| Boolean combinations conceal named commands or capabilities | Explicit commands or cooperative capability mixins, where combinations/order are domain-defined | User example; mechanism guide (not claimed as an exact PR flag rewrite) |
| Expected outcomes hidden behind catch routing; raw string statuses | Named outcomes/states where callers require that distinction | User example; mechanism guide (not claimed as an exact PR state rewrite) |
| `FooAuthority`/`FooDeclaration` wrappers simply forward data | Reuse the actual domain owner and remove redundant forwarding | User naming criterion; assess behavior, not suffix alone |

Read [nominal mechanisms](nominal-mechanisms.md) for the last three shapes and
Enum/state, validation, MI and metaclass semantics. The snippets below explicitly
separate runnable **teaching examples** from abbreviated **historical source**.
Teaching examples specify a tiny new domain; they are not production patches or
proofs that any arbitrary before-source is equivalent.

## 1. A declared set should determine its projections

**Bad teaching fragment:** every additional member requires another edit.

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

**Maintenance gain:** validation/display membership no longer requires remembering
a second roster. A static expected set in a test is an assertion, not a second
production authority. If a UI deliberately permits only CSV/JSON, that is a
separate selection policy, not automatically the full vocabulary's replica.

**Historical basis:** #60 `5a57218b3→63f0ede15`,
`openhcs/interop/cellprofiler/module_semantics.py`: independently authored family
and alias tuples gave way to projections over module declarations. The old
`MappingProxyType` maps were already derived: do not label them independent
mirrors. **NRA trajectory:** select the declaration and redundant roster, derive
its consumer projection using supported operations, migrate consumers, remove the
old writable roster; test new members, aliases and unknown values.

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

**Better — runnable teaching example:** consumer knows the public operation;
parent owns the invariant and leaf supplies only variation.

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

**Maintenance gain:** adding a transform supplies one hook, not another copied
validation sequence or a consumer-side branch. Construction/selection still needs
a defined boundary; this fragment intentionally does not claim to solve a wire
string→instance registry or unknown kind handling.

**Actual positive source:** #60 child `63f0ede15`,
`openhcs/processing/backends/cellprofiler/module_classes.py`,
`ModuleSettingsSourceModule.settings_source` is abstract while `bind_settings`
calls it and owns shared finalization. #38 child `05eb87f07^2`,
`core/memory/conversion_helpers.py`, `MemoryTypeConverter` supplies four abstract
hooks used by generated conversion orchestration. That after-source still has an
`_OPS` table and dynamic `eval`: learn the algorithm/hook decomposition, not the
claim that every table or effect disappeared. **NRA trajectory:** move shared
members to the admitted parent, migrate exact calls, preserve creation/binding
and all unknown/default paths, then check descendants and before/after effects.

## 3. A metaclass can derive registration from a class declaration

**Bad shape:** declare module classes, then separately hand-author all their
names and aliases in another roster, and runtime leaf policy names elsewhere.
A new class now demands synchronized updates.

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

**Maintenance gain:** the declared class carries the name/aliases that consumers
project. Shared validation belongs at the class-definition boundary, not in every
consumer. Reuse the project's registration mechanism rather than write another
metaclass. **NRA trajectory:** migrate declaration-owned facts first, derive
registry consumers, then remove old name/alias rosters only after finding all
uses. Check duplicate/inherited keys, imported/dynamic classes, aliases and unknown
names. Setup-only `SetupModuleCompiler` remains a separate family. #44's actual
`WidgetMeta` warns then overwrites duplicate IDs: automatic registration alone
is not a uniqueness guarantee.

## 4. Multiple inheritance composes real implementation capabilities

**Bad shape:** image-only, object-only and combined consumers separately copy
the same image/object input computation, or branch over the combinations.

**Runnable teaching example of the C3 mechanism:** these fixed input labels
stand for a tiny declared contract; production #60 derives artifacts from settings.

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

**Actual counterpart:** #60 child `63f0ede15` composes
`ScopedMeasurementModule(ImageMeasurementInputModule, ObjectMeasurementInputModule)`.
Both parents implement `measurement_artifact_inputs` cooperatively: image inputs
precede `super()` output; object inputs follow it. The base owns the shared
endpoint. **Maintenance gain:** one image-input implementation participates in
several legitimate combinations; it is not copied into each leaf. **NRA
trajectory:** establish the admitted bases/order, promote common implementations
and remove duplicates, prove selected member lookup and test output/effect order.
This example demonstrates one MRO, not interchangeable base orders or arbitrary
cooperative constructor correctness.

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

**Maintenance gain:** shared file/source/tabular fields are declared once; a
format chooses the applicable capabilities and adds only its own options.
`MaterializationSpec` explicitly rejects dicts and unregistered option types;
`materialize` selects `_WRITERS_BY_OPTIONS[type(opt)]`. Writer registration and
`MaterializationFormat` are still explicit, not inferred automatically from class
names. Dataclass annotations do not themselves validate field values. **NRA
trajectory:** introduce typed carriers and shared bases, migrate constructors and
exact callers, adapt the boundary, then remove old string/dict paths. Test new
format registration, default field ordering, malformed options and serialization.

## 6. Separate state lifetime from the view; derive instead of synchronizing

**Before shape:** the view owns mutable parameters/defaults/reset flags and
must synchronize other views or saved snapshots whenever it changes them.

**After #58, abridged source** (`43a6cba5b`, PFM):

```python
@property
def parameters(self):
    return self.state.parameters
```

The historical property is a direct projection. `ObjectState` owns parameters,
reset/user-set fields and mutation methods such as `update_parameter` and
`reset_parameter`; PFM takes `state` in its constructor. **Maintenance gain:**
view reopening need not recreate the model's identity, and a view reads the state
rather than maintaining another writable copy. **NRA trajectory:** establish the
state lifetime/construction boundary, retain the state object, migrate reads and
mutations, then delete redundant storage. Trace each actual caller: nested state
and separate scope-registry lookup paths are not proved identical by `self.state`.
The sixth dynamic `getattr` property and saved/live snapshots remain separate
obligations. Snapshot state can be deliberately independent, not a mirror to delete.

## Transfer the move, not the incidental spelling

For each application show: **old independent edit sites → new determining
operation/declaration → derived consumers → one new-case experiment**. Keep
NRA's [canonical pattern catalog](../../../docs/source/api/pattern_catalog.rst)
and registered DSL operations as the tool authorities; this cookbook is the
agent's judgment training, not a claim that every example has an automatic
recipe. See [ownership-to-DSL](ownership-to-dsl.md) for simulation/application gates.
