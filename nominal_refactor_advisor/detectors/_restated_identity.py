"""Restated identity: a string that spells a class, module, family member or enum value the scan declares."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from nominal_refactor_advisor.class_identity_observations import (
    DeclaredIdentities,
    DeclaredIdentityFamily,
    IdentitySpelling,
    IdentitySpellingFamily,
    ModuleDeclarations,
)
from nominal_refactor_advisor.detectors._base import (
    CompactProjectionGroups,
    DetectorConfig,
)
from nominal_refactor_advisor.scan_cache import ScanCache
from nominal_refactor_advisor.refactors.declaration_reference import (
    DeclarationReferenceFindingRecipeSynthesizer,
    ReferenceClassNameOperation,
    ReferenceEnumMemberOperation,
    ReferenceImportedModuleOperation,
    ReferenceLookedUpMemberOperation,
    UnquoteResolvedTypeNamesOperation,
)
from nominal_refactor_advisor.models import (
    CarriedReferenceRewrites,
    CoveredReferenceRewrites,
    RefactorFinding,
    SourceLocation,
    StoppedReferenceRewrites,
)
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.observation_shapes import RestatingSite
from nominal_refactor_advisor.taxonomy import CapabilityTag, ObservationTag
from nominal_refactor_advisor.detectors._restated_facts import (
    JoinedRestatedFactDetector,
    RestatedFact,
    _listed,
)


_CLASS_LITERAL_TAGS = frozenset(("type", "annotation", "word", "compared word", "class name"))


@dataclass(frozen=True)
class _RestatedIdentityBaseline:
    decl_by_path: dict[str, ModuleDeclarations]
    site_mod_by_name: dict[str, object]
    resolved_by_name: dict[str, tuple[RestatingSite, ...]]
    identities: DeclaredIdentities
    mods_by_class_lit: dict[str, tuple[str, ...]]
    mods_by_module_lit: dict[str, tuple[str, ...]]
    mods_by_member_attr: dict[str, tuple[str, ...]]
    mods_with_compared_word: frozenset[str]
    mods_by_lookup_method: dict[str, tuple[str, ...]]
    mods_by_enum_lit: dict[str, tuple[str, ...]]
    mods_with_enum_value: frozenset[str]

    @classmethod
    def build(
        cls,
        decl_modules: Sequence[ModuleDeclarations],
        site_modules: Sequence[object],
        resolved_by_name: dict[str, tuple[RestatingSite, ...]],
        identities: DeclaredIdentities,
    ) -> _RestatedIdentityBaseline:
        by_class_lit: dict[str, set[str]] = defaultdict(set)
        by_module_lit: dict[str, set[str]] = defaultdict(set)
        by_member_attr: dict[str, set[str]] = defaultdict(set)
        with_compared_word: set[str] = set()
        by_lookup_method: dict[str, set[str]] = defaultdict(set)
        by_enum_lit: dict[str, set[str]] = defaultdict(set)
        with_enum_value: set[str] = set()
        for sm in site_modules:
            mname = sm.module_name  # type: ignore[attr-defined]
            for site in sm.sites:  # type: ignore[attr-defined]
                tag = site.detail[0]
                if tag in _CLASS_LITERAL_TAGS:
                    by_class_lit[site.detail[-1]].add(mname)
                    if tag == "compared word":
                        with_compared_word.add(mname)
                elif tag == "module":
                    by_module_lit[site.detail[-1]].add(mname)
                elif tag in ("member", "flattened"):
                    by_member_attr[site.detail[1]].add(mname)
                elif tag == "lookup":
                    by_lookup_method[site.detail[2]].add(mname)
                elif tag == "enum value":
                    by_enum_lit[site.detail[-1]].add(mname)
                    with_enum_value.add(mname)
        return cls(
            decl_by_path={dm.file_path: dm for dm in decl_modules},
            site_mod_by_name={sm.module_name: sm for sm in site_modules},  # type: ignore[attr-defined]
            resolved_by_name=resolved_by_name,
            identities=identities,
            mods_by_class_lit={k: tuple(v) for k, v in by_class_lit.items()},
            mods_by_module_lit={k: tuple(v) for k, v in by_module_lit.items()},
            mods_by_member_attr={k: tuple(v) for k, v in by_member_attr.items()},
            mods_with_compared_word=frozenset(with_compared_word),
            mods_by_lookup_method={k: tuple(v) for k, v in by_lookup_method.items()},
            mods_by_enum_lit={k: tuple(v) for k, v in by_enum_lit.items()},
            mods_with_enum_value=frozenset(with_enum_value),
        )


@ScanCache.cached
def _base_restated_identity_slot() -> list[_RestatedIdentityBaseline | None]:
    return [None]


def _module_declarations_semantically_equal(
    old: ModuleDeclarations, new: ModuleDeclarations
) -> bool:
    if old is new:
        return True
    if (
        old.module_name != new.module_name
        or old.imports != new.imports
        or old.from_imports != new.from_imports
        or old.rebound != new.rebound
        or old.renamed != new.renamed
        or old.imported != new.imported
        or old.names != new.names
        or len(old.classes) != len(new.classes)
        or tuple(name for name, _line in old.bound_after)
        != tuple(name for name, _line in new.bound_after)
    ):
        return False
    return all(
        replace(a, line=0) == replace(b, line=0)
        for a, b in zip(old.classes, new.classes, strict=True)
    )


class RestatedIdentityDetector(
    JoinedRestatedFactDetector[DeclaredIdentities, RestatingSite],
    DeclarationReferenceFindingRecipeSynthesizer,
):
    """A string literal that spells a declared class, module, family member or enum value in a position that uses it."""

    module_projection_families = (IdentitySpellingFamily, DeclaredIdentityFamily)
    restated = RestatedFact(
        fact="The string spells a declared identity (a class name, a module's dotted name, a family member's distinct spelling, or an enum member's value), a second copy of it that a rename or a same-spelled sibling breaks silently.",
        owner="The owner is the declaration itself: the class object, the imported module, a method the family declares, or the enum member.",
        factoring="Reference the declaration instead of spelling it: name the class (an unquoted annotation, or the class object passed or compared), import the module and pass it or its `__name__`, replace a comparison of member spellings with the member class or a method the family declares, write a lookup by a member's spelling (`F.member_named('x')`, the spelling explicit or derived by the rule the repository declares in `derived-spellings`) as the member class, and write a string in a position a declaration types as an enum as the member it equals, `E.MEMBER` (rewritten when every base up to `StrEnum` keeps the member answering as its string); keep the spelling only in the family's boundary decoder.",
        operations=(
            UnquoteResolvedTypeNamesOperation,
            ReferenceImportedModuleOperation,
            ReferenceClassNameOperation,
            ReferenceEnumMemberOperation,
            ReferenceLookedUpMemberOperation,
        ),
    )
    finding_spec = restated.spec(
        PatternId.RESTATED_IDENTITY,
        "A string restates a declared identity",
        "the declared class, module, family member or enum member referenced directly instead of spelled",
        "a string literal equal to a class name, a scanned module's dotted name or a family member's distinct class keyword or constant, in a type, comparison, match, argument, return or importer position; or a string equal to a member's value of the enum a declaration types its position as (an annotated assignment or field, through `ClassVar`/`Final`, a parameter default, a return in a function declared `-> E`, or a comparison or match against a parameter declared `E` that its function never rebinds)",
        (CapabilityTag.NOMINAL_IDENTITY,),
        (ObservationTag.SEMANTIC_STRING_LITERAL,),
    )

    compact_shared_group_context_builder = staticmethod(
        DeclaredIdentities.from_projection_groups
    )

    def _findings_from_compact_projection_groups_context(
        self,
        projections_by_family: CompactProjectionGroups,
        context: DeclaredIdentities,
        config: DetectorConfig,
    ) -> list[RefactorFinding]:
        del config
        site_modules = projections_by_family[IdentitySpellingFamily]
        decl_modules = DeclaredIdentityFamily.modules_in(projections_by_family)
        if ScanCache.active() is not None:
            slot = _base_restated_identity_slot()
            base = slot[0]
            if base is not None:
                base_decl_by_path = base.decl_by_path
                base_site_mod_by_name = base.site_mod_by_name
                base_resolved_by_name = base.resolved_by_name
                decls_sem_equal = len(decl_modules) == len(base_decl_by_path)
                changed_decl_mods: list[ModuleDeclarations] = []
                if decls_sem_equal:
                    for dm in decl_modules:
                        old_dm = base_decl_by_path.get(dm.file_path)
                        if old_dm is dm:
                            continue
                        if old_dm is None or not _module_declarations_semantically_equal(
                            old_dm, dm
                        ):
                            decls_sem_equal = False
                            changed_decl_mods.append(dm)
                else:
                    for dm in decl_modules:
                        old_dm = base_decl_by_path.get(dm.file_path)
                        if old_dm is not dm:
                            changed_decl_mods.append(dm)
                if decls_sem_equal:
                    resolved: list[RestatingSite] = []
                    for sm in site_modules:
                        old_sm = base_site_mod_by_name.get(sm.module_name)
                        if old_sm is sm:
                            cached_sites = base_resolved_by_name.get(sm.module_name)
                            if cached_sites:
                                resolved.extend(cached_sites)
                        else:
                            for site in sm.sites:
                                if (found := context.resolve(site)) is not None:
                                    resolved.append(found)
                    return super().findings_from_sites(
                        resolved, context.with_carriers(resolved)
                    )
                base_id = base.identities
                dirty_site_mods: set[str] = set()
                for sm in site_modules:
                    if base_site_mod_by_name.get(sm.module_name) is not sm:
                        dirty_site_mods.add(sm.module_name)
                changed_classes = {
                    k
                    for k in (base_id.classes.keys() | context.classes.keys())
                    if base_id.classes.get(k) != context.classes.get(k)
                }
                for cname in changed_classes:
                    dirty_site_mods.update(base.mods_by_class_lit.get(cname, ()))
                changed_modules = {
                    k
                    for k in (base_id.modules.keys() | context.modules.keys())
                    if base_id.modules.get(k) != context.modules.get(k)
                }
                for mname in changed_modules:
                    dirty_site_mods.update(base.mods_by_module_lit.get(mname, ()))
                changed_members = {
                    k
                    for k in (base_id.members.keys() | context.members.keys())
                    if base_id.members.get(k) != context.members.get(k)
                }
                bases_changed = base_id.bases != context.bases
                inst_changed = (
                    base_id.instance_attributes != context.instance_attributes
                )
                if changed_members:
                    dirty_site_mods.update(base.mods_with_compared_word)
                    for attr in changed_members:
                        dirty_site_mods.update(base.mods_by_member_attr.get(attr, ()))
                if changed_classes or bases_changed or inst_changed:
                    for attr in base_id.members.keys() | context.members.keys():
                        dirty_site_mods.update(base.mods_by_member_attr.get(attr, ()))
                lookups_changed = base_id.lookups != context.lookups
                if changed_classes or changed_members or bases_changed or lookups_changed:
                    for method in base_id.all_lookup_methods | context.all_lookup_methods:
                        dirty_site_mods.update(base.mods_by_lookup_method.get(method, ()))
                changed_enum_lits = {
                    k
                    for k in (base_id.enum_values.keys() | context.enum_values.keys())
                    if base_id.enum_values.get(k) != context.enum_values.get(k)
                }
                for lit in changed_enum_lits:
                    dirty_site_mods.update(base.mods_by_enum_lit.get(lit, ()))
                if changed_decl_mods or len(decl_modules) != len(base_decl_by_path):
                    for dm in changed_decl_mods:
                        if dm.module_name in base.mods_with_enum_value:
                            dirty_site_mods.add(dm.module_name)
                    # If any changed/removed module could alter enum canonical imports, re-check
                    # modules whose enum-value literals exist in old or new enum_values.
                    for lit in base_id.enum_values.keys() | context.enum_values.keys():
                        dirty_site_mods.update(base.mods_by_enum_lit.get(lit, ()))
                resolved = []
                for sm in site_modules:
                    if sm.module_name not in dirty_site_mods:
                        cached_sites = base_resolved_by_name.get(sm.module_name)
                        if cached_sites:
                            resolved.extend(cached_sites)
                    else:
                        for site in sm.sites:
                            if (found := context.resolve(site)) is not None:
                                resolved.append(found)
                return super().findings_from_sites(
                    resolved, context.with_carriers(resolved)
                )
            resolved_by_name: dict[str, tuple[RestatingSite, ...]] = {}
            all_resolved: list[RestatingSite] = []
            for sm in site_modules:
                mod_resolved = tuple(
                    found
                    for site in sm.sites
                    if (found := context.resolve(site)) is not None
                )
                if mod_resolved:
                    resolved_by_name[sm.module_name] = mod_resolved
                    all_resolved.extend(mod_resolved)
            if slot[0] is None:
                slot[0] = _RestatedIdentityBaseline.build(
                    decl_modules, site_modules, resolved_by_name, context
                )
            return super().findings_from_sites(
                all_resolved, context.with_carriers(all_resolved)
            )
        return super()._findings_from_compact_projection_groups_context(
            projections_by_family, context, DetectorConfig()
        )

    def findings_from_sites(
        self, sites: Iterable[RestatingSite], context: DeclaredIdentities
    ) -> list[RefactorFinding]:
        resolved = [
            found for site in sites if (found := context.resolve(site)) is not None
        ]
        return super().findings_from_sites(resolved, context.with_carriers(resolved))

    def describe(
        self, sites: Sequence[RestatingSite], context: DeclaredIdentities
    ) -> str:
        spelled = [
            f"{IdentitySpelling.positions()[site.detail[0]].spelling(site.detail)} "
            f"({site.detail[0]}, line {site.lineno})"
            for site in sites
        ]
        return (
            f"{sites[0].symbol} spells {sites[0].group} {len(sites)} time(s): "
            f"{_listed(spelled)}."
        )

    def owner(self, sites: Sequence[RestatingSite], context: DeclaredIdentities) -> str:
        declared = context.declarations_of(sites)
        if not declared:
            return self.restated.owner
        where = _listed(
            f"`{location.symbol}` ({location.file_path}:{location.line})"
            for location in declared
        )
        return f"{self.restated.owner} Declared at {where}."

    def evidence(
        self, sites: Sequence[RestatingSite], context: DeclaredIdentities
    ) -> tuple[SourceLocation, ...]:
        return (*super().evidence(sites, context), *context.declarations_of(sites))

    def finding_context(
        self, sites: Sequence[RestatingSite], context: DeclaredIdentities
    ) -> dict[str, object]:
        carried = context.carried_rewrites(sites)
        if carried:
            return {"metrics": CarriedReferenceRewrites(carried)}
        covering = context.covering_rewrites(sites)
        if covering:
            return {"metrics": CoveredReferenceRewrites(covering)}
        site_kinds = tuple(sorted({site.detail[0] for site in sites}))
        if site_kinds:
            return {"metrics": StoppedReferenceRewrites(site_kinds)}
        return {}


__all__ = ("RestatedIdentityDetector",)
