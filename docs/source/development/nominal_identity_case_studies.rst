Nominal Identity Case Studies
=============================

Concrete guidance for agents and maintainers on when nominal identity is required, why duck typing
cannot recover the same capabilities, and how the mathematical models from the two papers apply to
real OpenHCS patterns.

Read ``nominal_architecture_playbook.rst`` first for the unified model, then use this document for
worked examples.

This document uses the paper-level vocabulary directly:

- a structural view is a partial representation of an object
- a fiber is the set of semantically distinct objects that collapse to the same visible structure
- a confusability gap appears when the architecture must distinguish cases that the current view cannot
- a nominal handle is the explicit identity mechanism that separates those cases

The core rule is simple:

If the system needs enumeration, exhaustiveness, provenance, conflict ordering, class-level registration,
or O(1) lookup by role, then structure alone is not enough. The code needs nominal identity.


Why This Matters
----------------

Paper 1 shows that equal visible structure does not recover semantic identity. Paper 2 shows that
verifiable integrity requires both a single authoritative source and observable provenance. Together,
they imply the following software rule:

- duck typing is a partial view
- nominal typing supplies the missing identity handle
- sentinel attributes can simulate identity by convention, but they do not recover the capabilities that
  come from declared type identity

In Python, those capabilities include:

- ``__subclasses__()`` enumeration
- ``isinstance()`` and ``issubclass()`` checks
- MRO-based conflict resolution
- metaclass registration
- class-level markers
- dynamic interface generation
- type-keyed dictionaries
- method injection into a shared type namespace


Corollary 5.2b: Specific Sentinel Failures
------------------------------------------

Sentinel attributes are often proposed as a substitute for nominal identity:

.. code-block:: python

   class SomeMixin:
       sigma = "step"

The papers say this only simulates identity at the representation level. It does not recover the
capabilities that actual type identity provides.

.. list-table:: Sentinel capability failures
   :header-rows: 1

   * - Capability
     - Why a sentinel attribute fails
   * - Enumeration
     - Enumeration requires iterating over all candidate types with ``sigma == v``. Structural typing has
       no built-in type registry, so there is no principled source for the search space.
   * - Enforcement
     - ``sigma`` is just a runtime value, not a type constraint. Subclasses can set it incorrectly without
       any type error or import-time failure.
   * - Conflict resolution
     - If multiple mixins define ``sigma``, the system needs MRO ordering to decide which definition wins.
       A bare sentinel value has no MRO.
   * - Provenance
     - Answering ``which type provided sigma?`` requires class hierarchy traversal. The sentinel value
       cannot explain its own origin.


Corollary 5.2c: Sentinel Simulates, Cannot Recover
--------------------------------------------------

Sentinel attributes can simulate type identity by convention, but they cannot recover the capabilities
that identity provides. The simulation is:

- unenforced: wrong values do not fail loudly
- unenumerable: there is no built-in registry of all structurally matching types
- unordered: there is no MRO for conflicts among plain sentinel values

Agent rule:

Do not replace a nominal type boundary with a sentinel attribute when the surrounding code depends on
enumeration, ordering, provenance, or exhaustive dispatch.

The same MRO logic also means that orthogonal reusable concerns often belong in mixins or multiple
inheritance, not in composition wrappers that erase precedence and nominal participation.


Case Study 2: Discriminated Unions via ``__subclasses__()``
-----------------------------------------------------------

OpenHCS parameter UIs need exhaustive widget creation dispatch based on parameter type structure.

.. code-block:: python

   @dataclass
   class OptionalDataclassInfo(ParameterInfoBase):
       widget_creation_type: str = "OPTIONAL_NESTED"

       @staticmethod
       def matches(param_type: Type) -> bool:
           return is_optional(param_type) and is_dataclass(inner_type(param_type))


   @dataclass
   class DirectDataclassInfo(ParameterInfoBase):
       widget_creation_type: str = "NESTED"

       @staticmethod
       def matches(param_type: Type) -> bool:
           return is_dataclass(param_type)


   @dataclass
   class GenericInfo(ParameterInfoBase):
       @staticmethod
       def matches(param_type: Type) -> bool:
           return True

The factory enumerates ``ParameterInfoBase.__subclasses__()`` and therefore gets runtime exhaustiveness
for free. Adding ``EnumInfo`` or another variant extends the dispatch family without modifying the
factory.

Why nominal identity matters:

- inheritance gives a built-in registry of all declared variants
- the registry is refactoring-safe because it is keyed by type identity, not strings or method names
- the dispatch can do one linear scan over known variants and then dispatch directly

Why duck typing fails:

- there is no query equivalent to ``what are all types that implement matches()?``
- structural typing would require a hand-maintained registry list
- the list becomes a second writable source and can drift

Agent guidance:

When the code needs exhaustive variant enumeration, prefer subclass families over duck-typed method
probing.


Case Study 3: MemoryTypeConverter Dispatch
------------------------------------------

The converter layer generates one concrete converter per memory backend at module load.

.. code-block:: python

   _CONVERTERS = {
       mem_type: type(
           f"{mem_type.value.capitalize()}Converter",
           (MemoryTypeConverter,),
           _TYPE_OPERATIONS[mem_type],
       )()
       for mem_type in MemoryType
   }


   def convert_memory(data, source_type: str, target_type: str, gpu_id: int):
       source_enum = MemoryType(source_type)
       converter = _CONVERTERS[source_enum]
       method = getattr(converter, f"to_{target_type}")
       return method(data, gpu_id)

All converters share the same visible method family but have different implementations.

Why nominal identity matters:

- the generated types are distinct identities even when their signatures are similar
- the dispatch table can do O(1) lookup by declared backend type
- the architecture does not need to probe every candidate converter

Why duck typing fails:

- structurally, all converters look alike
- the code would need repeated ``hasattr`` probing or a parallel string registry
- both approaches collapse semantically distinct converters into the same structural fiber

Agent guidance:

When the system already has a closed backend enum and distinct backend implementations, use a nominal
converter family plus type- or enum-keyed O(1) dispatch.


Case Study 4: Polymorphic Configuration
---------------------------------------

The streaming subsystem uses distinct viewer configs with explicit backend semantics.

.. code-block:: python

   class StreamingConfig(StreamingDefaults, ABC):
       @property
       @abstractmethod
       def backend(self) -> Backend:
           raise NotImplementedError


   NapariStreamingConfig = create_streaming_config(
       viewer_name="napari", port=5555, backend=Backend.NAPARI_STREAM
   )
   FijiStreamingConfig = create_streaming_config(
       viewer_name="fiji", port=5565, backend=Backend.FIJI_STREAM
   )

The codebase explicitly prefers:

- old: ``hasattr(config, 'napari_port')``
- new: ``isinstance(config, NapariStreamingConfig)``

Why nominal identity matters:

- the check is tied to declared role identity, not fragile field names
- renaming a field does not silently break dispatch
- the ABC gives fail-loud validation at definition time

Why duck typing fails:

- attribute-name checks couple behavior to strings
- field renames break call sites silently
- structurally similar configs become confusable even when they represent different backends

Agent guidance:

If the dispatch question is really ``which config family is this?``, use an ABC and concrete subclasses,
not stringly attribute checks.


Case Study 5: Migration from Duck Typing to ABC Contracts
---------------------------------------------------------

PR #44 migrated the UI layer from scattered duck typing to centralized nominal contracts.

Before:

- ``ParameterFormManager`` had 47 ``hasattr()`` dispatch points
- mixins probed widget capabilities by attribute names
- dispatch tables mapped string attribute names to handlers

After:

- one ``AbstractFormWidget`` ABC declared the contract
- call sites used ``isinstance()`` and direct method calls
- repeated probing logic disappeared

.. code-block:: python

   # BEFORE
   if hasattr(widget, "isChecked"):
       return widget.isChecked()
   elif hasattr(widget, "currentText"):
       return widget.currentText()


   # AFTER
   class AbstractFormWidget(ABC):
       @abstractmethod
       def get_value(self) -> Any:
           raise NotImplementedError

Why nominal identity matters:

- the contract moves to one authoritative declaration point
- typos and missing methods fail loudly when classes are defined
- the architecture becomes inspectable and exhaustively extensible

Why duck typing fails:

- error detection shifts to user interaction time
- missing attributes can silently produce ``None`` or fall through the wrong branch
- the same semantic role is rediscovered at dozens of call sites

Agent guidance:

When you see dozens of ``hasattr`` branches for the same role family, that is not flexibility. It is a
signal to introduce an ABC and move the role definition into one place.


Case Study 6: AutoRegisterMeta
------------------------------

Metaclass-based auto-registration depends on type identity as the registration value.

.. code-block:: python

   class AutoRegisterMeta(ABCMeta):
       def __new__(mcs, name, bases, attrs, registry_config=None):
           new_class = super().__new__(mcs, name, bases, attrs)
           if getattr(new_class, "__abstractmethods__", None):
               return new_class
           key = mcs._get_registration_key(name, new_class, registry_config)
           registry_config.registry_dict[key] = new_class
           return new_class

Why nominal identity matters:

- the registry stores classes as distinct first-class objects
- abstract classes can be skipped by inspecting class-level metadata
- key derivation can depend on class names and inheritance structure

Why duck typing fails:

- structural probing is instance-centric and cannot naturally represent ``the class itself`` as the value
- there is no principled analogue of skipping abstract classes via ``__abstractmethods__``
- duplicates are harder to reject at import time because there is no declared identity boundary

Agent guidance:

Do not refactor metaclass registration patterns toward instance-level duck typing. They are inherently
class-level nominal mechanisms.


Case Study 7: Five-Stage Type Transformation
--------------------------------------------

The decorator chain around global config types performs systematic type transformation with lineage
tracking.

Stage 1
~~~~~~~

``@auto_create_decorator`` validates naming, marks the class, creates a companion decorator, and exports
it at module scope.

Stage 2
~~~~~~~

The generated decorator rebuilds nested configs, generates lazy companions, exports the lazy class, and
registers pending field injection.

Stage 3
~~~~~~~

``LazyDataclassFactory.make_lazy_simple()`` creates a new lazy type and registers the lazy-to-base and
base-to-lazy mapping.

Stage 4
~~~~~~~

Pending configs are injected back into the global config type as lazy fields.

Stage 5
~~~~~~~

Runtime resolution walks ``type(config).__mro__`` and scope context together, returning provenance as
``(value, scope, source_type)``.

Why nominal identity matters:

- each generated type has lineage that must remain distinguishable
- the registries depend on exact type identity, not mere field equality
- MRO traversal depends on class hierarchy, not structural coincidence

Why duck typing fails:

- structurally equivalent lazy and base types become indistinguishable
- lineage tracking collapses because there is no stable identity to map from and to
- scope x MRO resolution loses meaning if type identity is erased

Agent guidance:

If the architecture creates companion types, lazy variants, rebuilt classes, or decorator-generated type
families, preserve the nominal lineage. Do not collapse them into structural equivalence classes.


Case Study 8: Dual-Axis Resolution Algorithm
--------------------------------------------

The inheritance resolver walks scope hierarchy and class hierarchy together.

.. code-block:: python

   def resolve_field_inheritance(obj, field_name, scope_stack):
       mro = [normalize_type(T) for T in type(obj).__mro__]
       for scope in scope_stack:
           for mro_type in mro:
               config = get_config_at_scope(scope, mro_type)
               if config and hasattr(config, field_name):
                   value = getattr(config, field_name)
                   if value is not None:
                       return (value, scope, mro_type)
       return (None, None, None)

Why nominal identity matters:

- the algorithm returns provenance as ``mro_type``
- earlier positions in the MRO encode override priority
- the search space is the ordered product of scope hierarchy and class hierarchy

Why duck typing fails:

- structurally similar config types provide no principled ordering axis
- provenance becomes meaningless if two types are treated as identical because their fields match
- the algorithm degenerates into ad hoc sequential probing with no explicit conflict rule

Agent guidance:

When a resolver returns ``which type supplied the answer``, nominal identity is part of the result, not an
implementation detail.


Case Study 9: Custom ``isinstance()``
-------------------------------------

Virtual inheritance uses a metaclass to make class-level markers participate in ``isinstance()``.

.. code-block:: python

   class GlobalConfigMeta(type):
       def __instancecheck__(cls, instance):
           if hasattr(instance.__class__, "_is_global_config"):
               return instance.__class__._is_global_config
           return super().__instancecheck__(instance)

Why nominal identity matters:

- the question is class-level: ``does this type claim this role?``
- the marker lives on the class, not on arbitrary instances
- the metaobject protocol is explicitly operating on type identity

Why duck typing fails:

- instance-level ``hasattr(instance, ...)`` is not the same as class-level interface membership
- there is no robust way to say ``this class explicitly implements this runtime-checkable interface``
  without nominal machinery

Agent guidance:

Do not confuse virtual inheritance or class-level markers with ordinary duck typing. They are nominal
mechanisms expressed through the metaclass layer.


Case Study 10: Dynamic Interface Generation
-------------------------------------------

Runtime-generated interfaces can exist purely for nominal identity.

.. code-block:: python

   class DynamicInterfaceMeta(ABCMeta):
       _generated_interfaces: Dict[str, Type] = {}

       @classmethod
       def get_or_create_interface(mcs, interface_name: str) -> Type:
           if interface_name not in mcs._generated_interfaces:
               interface = type(interface_name, (ABC,), {})
               mcs._generated_interfaces[interface_name] = interface
           return mcs._generated_interfaces[interface_name]

The generated ABCs may have empty namespaces. Their whole purpose is explicit identity.

Why nominal identity matters:

- ``IStreamingConfig`` and ``IVideoConfig`` can be distinct even with no structural content
- interface membership is a declared claim, not an emergent property of current attributes

Why duck typing fails:

- structurally empty interfaces collapse to ``object`` under structural reasoning
- there is no way to express explicit role membership without probing for fields that do not exist

Agent guidance:

Empty ABCs are not automatically useless. If they carry explicit role identity, they are acting as nominal
handles.


Case Study 11: Framework Detection via Sentinel Type
----------------------------------------------------

A runtime-generated sentinel object can act as a unique nominal marker.

.. code-block:: python

   _FRAMEWORK_CONFIG = type("_FrameworkConfigSentinel", (), {})()


   def has_framework_config():
       return _FRAMEWORK_CONFIG in GlobalRegistry.configs

Why nominal identity matters:

- the sentinel's identity is unique and refactoring-safe
- the check is decoupled from module names, attribute names, or string keys
- another module creating the same-looking sentinel still gets a distinct identity

Why duck typing fails:

- string keys are not type-safe
- module probing is fragile and tied to implementation details
- structural equality cannot guarantee uniqueness across independently created sentinels

Agent guidance:

If the requirement is ``presence of this exact capability marker``, a nominal sentinel type is more robust
than strings or attribute-name probing.


Case Study 12: Dynamic Method Injection
---------------------------------------

Method injection modifies a shared type namespace.

.. code-block:: python

   def inject_conversion_methods(target_type: Type, methods: Dict[str, Callable]):
       for method_name, method_impl in methods.items():
           setattr(target_type, method_name, method_impl)

Why nominal identity matters:

- the target type is a first-class mutable namespace
- changing the type affects all current and future instances through method lookup
- the operation is defined on the class object itself

Why duck typing fails:

- instance-level mutation does not express ``modify the behavior of all objects of this kind``
- structural typing has no equivalent of a shared type namespace distinct from instance state

Agent guidance:

If the code is injecting methods, monkey-patching classes, or extending plugin types, do not describe the
pattern as duck typing. It is operating on nominal class objects.


Case Study 13: Bidirectional Type Lookup
----------------------------------------

OpenHCS maintains lazy-to-base and base-to-lazy registries using type identity as the key on both sides.

.. code-block:: python

   class BidirectionalTypeRegistry:
       def __init__(self):
           self._forward: Dict[Type, Type] = {}
           self._reverse: Dict[Type, Type] = {}

       def register(self, lazy_type: Type, base_type: Type):
           if lazy_type in self._forward:
               raise ValueError(f"{lazy_type} already registered")
           if base_type in self._reverse:
               raise ValueError(f"{base_type} already has lazy companion")
           self._forward[lazy_type] = base_type
           self._reverse[base_type] = lazy_type

Why nominal identity matters:

- type identity is a refactoring-safe key
- bijection violations fail loudly
- lookups are O(1) in both directions

Why duck typing fails:

- string keys introduce rename fragility
- parallel registries can drift out of sync
- structural similarity does not give a stable key for exact bidirectional lookup

Agent guidance:

When the code needs synchronized two-way registries, type identity is often the right authoritative key.


Agent Checklist
---------------

When reviewing or refactoring code, ask these questions in order:

1. Is the current decision procedure operating on full semantic identity, or only on a partial structural
   view?
2. If it is a partial view, which semantically distinct cases collapse into the same fiber?
3. Does the code need any of the capabilities that structural typing cannot recover?

   - exhaustive enumeration
   - import-time enforcement
   - MRO conflict ordering
   - provenance reporting
   - type-keyed O(1) lookup
   - metaclass registration
   - class-level markers
   - shared type namespaces

4. If yes, is the nominal handle already present and merely underused, or does the architecture need an
   explicit ABC / subclass / sentinel type / metaclass registry?
5. Can the duplicated machinery now be moved into the ABC or authoritative type declaration?


Refactoring Rules
-----------------

Prefer nominal identity when:

- the code must answer ``what exact role family is this?``
- the system needs exhaustive variant discovery
- provenance or origin of a value must be reported
- conflict resolution depends on inheritance order
- two-way registries or O(1) type-keyed dispatch are required
- decorators, metaclasses, or factories generate and track type lineages

Do not replace nominal identity with:

- ``Protocol`` used as a semantic role boundary
- sentinel attributes used as fake interface membership
- string keys where type keys are the real invariant
- repeated ``hasattr`` probing to rediscover a role already known to the architecture

Use duck typing only when the system truly needs none of the capabilities above and the role question is
purely behavioral, local, and non-exhaustive.


Final Rule
----------

Sentinels, strings, and structural probes can simulate identity at the representation level. They cannot
recover the architectural capabilities that explicit nominal identity provides.

When the code needs those capabilities, use the nominal tool directly.
