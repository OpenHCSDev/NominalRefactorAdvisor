from __future__ import annotations

from dataclasses import replace

import pytest

from nominal_refactor_advisor.relation_factorization import (
    CERTIFICATE_SCHEMA,
    COVER_TIE_BREAK,
    LEXICAL_TIE_BREAK,
    PARENT_TIE_BREAK,
    ExactSearchLimitExceeded,
    ExactSearchLimits,
    FormalConceptSearchStatistics,
    LaminarSearchStatistics,
    OwnerInvalidationSurface,
    ProviderCoverSearchStatistics,
    ProviderRectangle,
    RelationFactorizationCertificate,
    RequiredRelation,
    RequiredRelationPair,
    SingleParentArrangement,
    factor_laminar_relation,
    factor_provider_cover,
)


def relation(
    implementations: tuple[str, ...],
    consumers: tuple[str, ...],
    pairs: tuple[tuple[str, str], ...],
) -> RequiredRelation:
    return RequiredRelation.from_pairs(implementations, consumers, pairs)


def crossing_relation() -> RequiredRelation:
    return relation(
        ("left", "right", "zero"),
        ("a", "b", "c"),
        (("left", "a"), ("left", "b"), ("right", "b"), ("right", "c")),
    )


def assert_rejected_arrangement(
    retained: RequiredRelation,
    implementation_parents: tuple[tuple[str, str | None], ...],
    consumer_parents: tuple[tuple[str, str | None], ...],
    *,
    tie_break: str = PARENT_TIE_BREAK,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        SingleParentArrangement(
            implementation_parents=implementation_parents,
            consumer_parents=consumer_parents,
            tie_break=tie_break,
            retained=retained,
        )


def test_relation_normalizes_iteration_order_and_preserves_zero_degree_carriers() -> (
    None
):
    first = relation(
        ("zero", "beta", "alpha"),
        ("z", "x", "y"),
        (("beta", "z"), ("alpha", "y"), ("alpha", "x")),
    )
    second = relation(
        ("alpha", "beta", "zero"),
        ("x", "y", "z"),
        (("alpha", "x"), ("alpha", "y"), ("beta", "z")),
    )
    assert first == second
    assert first.implementation_keys == ("alpha", "beta", "zero")
    assert first.consumers_for("zero") == frozenset()
    assert first.implementations_for("z") == frozenset({"beta"})


@pytest.mark.parametrize(
    ("implementations", "consumers", "pairs"),
    [
        (("a", "a"), ("x",), ()),
        (("a",), ("x", "x"), ()),
        (("a",), ("x",), (("a", "x"), ("a", "x"))),
        (("a",), ("x",), (("foreign", "x"),)),
        (("a",), ("x",), (("a", "foreign"),)),
        (("",), ("x",), ()),
        ((" a",), ("x",), ()),
        (("a",), ("x ",), ()),
        (("e\u0301",), ("x",), ()),
    ],
)
def test_relation_rejects_duplicate_invalid_or_foreign_input(
    implementations: tuple[str, ...],
    consumers: tuple[str, ...],
    pairs: tuple[tuple[str, str], ...],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        relation(implementations, consumers, pairs)


def test_pair_is_exact_two_field_canonical_vocabulary() -> None:
    pair = RequiredRelationPair("implementation", "consumer")
    assert tuple(pair.__dataclass_fields__) == ("implementation_key", "consumer")
    assert pair.implementation_key == "implementation"
    assert pair.consumer == "consumer"


def test_exact_deletion_only_laminar_repair_crossing_and_statistics() -> None:
    required = crossing_relation()
    certificate = factor_laminar_relation(required)
    certificate.verify()
    assert certificate.tie_break == LEXICAL_TIE_BREAK
    assert len(certificate.retained.pairs) == 3
    assert len(certificate.residual.pairs) == 1
    assert certificate.ancestry_gap == certificate.residual.pairs
    assert certificate.statistics.subsets_examined == 16
    assert certificate.statistics.feasible_laminar_subsets > 0
    assert certificate.optimal_repair_count == certificate.statistics.optimal_subsets
    assert set(certificate.retained.pairs) | set(certificate.residual.pairs) == set(
        required.pairs
    )
    assert not (set(certificate.retained.pairs) & set(certificate.residual.pairs))


def test_nested_and_empty_scopes_have_canonical_single_parents() -> None:
    retained = relation(
        ("empty", "inner", "outer", "same"),
        ("a", "b"),
        (
            ("inner", "a"),
            ("outer", "a"),
            ("outer", "b"),
            ("same", "a"),
        ),
    )
    arrangement = SingleParentArrangement.from_relation(retained)
    assert arrangement.tie_break == PARENT_TIE_BREAK
    assert tuple(child for child, _ in arrangement.implementation_parents) == (
        "empty",
        "inner",
        "outer",
        "same",
    )
    for key in retained.implementation_keys:
        assert arrangement.descendants_for(key) == retained.consumers_for(key)
    assert dict(arrangement.consumer_parents)["a"] in {"inner", "same"}


def test_arrangement_validation_context_is_not_stored_or_serialized() -> None:
    retained = relation(("owner",), ("x",), (("owner", "x"),))
    arrangement = SingleParentArrangement.from_relation(retained)
    assert "retained" not in arrangement.__dict__
    assert tuple(arrangement.__dataclass_fields__) == (
        "implementation_parents",
        "consumer_parents",
        "tie_break",
        "retained",
    )


def test_arrangement_rejects_duplicate_missing_foreign_and_noncanonical_maps() -> None:
    retained = relation(
        ("child", "parent"),
        ("x", "y"),
        (("child", "x"), ("parent", "x"), ("parent", "y")),
    )
    valid = SingleParentArrangement.from_relation(retained)
    assert_rejected_arrangement(
        retained,
        (("child", "parent"), ("child", None)),
        valid.consumer_parents,
    )
    assert_rejected_arrangement(
        retained,
        (("child", "parent"),),
        valid.consumer_parents,
    )
    assert_rejected_arrangement(
        retained,
        (("child", "foreign"), ("parent", None)),
        valid.consumer_parents,
    )
    assert_rejected_arrangement(
        retained,
        tuple(reversed(valid.implementation_parents)),
        valid.consumer_parents,
    )
    assert_rejected_arrangement(
        retained,
        valid.implementation_parents,
        (("x", "child"), ("x", "parent")),
    )


def test_arrangement_rejects_cycles_wrong_policies_and_wrong_tie_break() -> None:
    retained = relation(
        ("child", "parent"),
        ("x", "y"),
        (("child", "x"), ("parent", "x"), ("parent", "y")),
    )
    valid = SingleParentArrangement.from_relation(retained)
    assert_rejected_arrangement(
        retained,
        (("child", "parent"), ("parent", "child")),
        valid.consumer_parents,
    )
    assert_rejected_arrangement(
        retained,
        (("child", None), ("parent", None)),
        valid.consumer_parents,
    )
    assert_rejected_arrangement(
        retained,
        valid.implementation_parents,
        (("x", "parent"), ("y", "parent")),
    )
    assert_rejected_arrangement(
        retained,
        valid.implementation_parents,
        valid.consumer_parents,
        tie_break="other",
    )


def test_arrangement_rejects_remaining_context_carrier_and_row_contracts() -> None:
    retained = relation(
        ("child", "parent"),
        ("x", "y"),
        (("child", "x"), ("parent", "x"), ("parent", "y")),
    )
    valid = SingleParentArrangement.from_relation(retained)
    assert_rejected_arrangement(
        retained,
        (("child", "parent"), ("parent", None)),
        (("x", "foreign"), ("y", "parent")),
    )
    assert_rejected_arrangement(
        retained,
        valid.implementation_parents,
        (("foreign", "child"), ("y", "parent")),
    )
    assert_rejected_arrangement(
        retained,
        valid.implementation_parents,
        (("x", "child"),),
    )
    assert_rejected_arrangement(
        retained,
        (("child", "child"), ("parent", None)),
        valid.consumer_parents,
    )
    with pytest.raises(TypeError):
        SingleParentArrangement(
            implementation_parents=valid.implementation_parents,
            consumer_parents=valid.consumer_parents,
            tie_break=valid.tie_break,
            retained="not-a-relation",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        SingleParentArrangement(
            implementation_parents=(("child",), ("parent", None)),  # type: ignore[arg-type]
            consumer_parents=valid.consumer_parents,
            tie_break=valid.tie_break,
            retained=retained,
        )
    with pytest.raises(TypeError):
        SingleParentArrangement(
            implementation_parents=(
                ["child", "parent"],  # type: ignore[list-item]
                ("parent", None),
            ),
            consumer_parents=valid.consumer_parents,
            tie_break=valid.tie_break,
            retained=retained,
        )

    class TupleSubclass(tuple[object, ...]):
        pass

    with pytest.raises(TypeError):
        SingleParentArrangement(
            implementation_parents=(
                TupleSubclass(("child", "parent")),  # type: ignore[arg-type]
                ("parent", None),
            ),
            consumer_parents=valid.consumer_parents,
            tie_break=valid.tie_break,
            retained=retained,
        )
    with pytest.raises(TypeError):
        SingleParentArrangement(
            implementation_parents=valid.implementation_parents,
            consumer_parents=(
                TupleSubclass(("x", "child")),  # type: ignore[arg-type]
                ("y", "parent"),
            ),
            tie_break=valid.tie_break,
            retained=retained,
        )
    with pytest.raises(TypeError):
        SingleParentArrangement(
            implementation_parents=[  # type: ignore[arg-type]
                ("child", "parent"),
                ("parent", None),
            ],
            consumer_parents=valid.consumer_parents,
            tie_break=valid.tie_break,
            retained=retained,
        )
    with pytest.raises(TypeError):
        SingleParentArrangement(
            implementation_parents=(  # type: ignore[arg-type]
                row for row in valid.implementation_parents
            ),
            consumer_parents=valid.consumer_parents,
            tie_break=valid.tie_break,
            retained=retained,
        )
    with pytest.raises(TypeError):
        SingleParentArrangement(
            implementation_parents=valid.implementation_parents,
            consumer_parents=[  # type: ignore[arg-type]
                ("x", "child"),
                ("y", "parent"),
            ],
            tie_break=valid.tie_break,
            retained=retained,
        )
    with pytest.raises(TypeError):
        SingleParentArrangement(
            implementation_parents=valid.implementation_parents,
            consumer_parents=(  # type: ignore[arg-type]
                row for row in valid.consumer_parents
            ),
            tie_break=valid.tie_break,
            retained=retained,
        )


def test_arrangement_rejects_nonlaminar_descendant_scope_mismatch() -> None:
    crossing = relation(
        ("left", "right"),
        ("a", "b", "c"),
        (("left", "a"), ("left", "b"), ("right", "b"), ("right", "c")),
    )
    assert_rejected_arrangement(
        crossing,
        (("left", None), ("right", None)),
        (("a", "left"), ("b", "left"), ("c", "right")),
    )


def test_all_64_two_owner_scope_pairs_reconstruct_exact_retained_descendants() -> None:
    consumers = ("a", "b", "c")
    for left_mask in range(8):
        for right_mask in range(8):
            required = relation(
                ("left", "right"),
                consumers,
                tuple(
                    (owner, consumer)
                    for owner, mask in (("left", left_mask), ("right", right_mask))
                    for index, consumer in enumerate(consumers)
                    if mask & (1 << index)
                ),
            )
            repair = factor_laminar_relation(required)
            repair.verify()
            for key in required.implementation_keys:
                assert repair.arrangement.descendants_for(key) == (
                    repair.retained.consumers_for(key)
                )


def test_provider_candidates_are_exact_maximal_bicliques_and_cover_residual() -> None:
    residual = relation(
        ("a", "b", "c"),
        ("x", "y", "z", "unused"),
        (
            ("a", "x"),
            ("a", "y"),
            ("b", "x"),
            ("b", "y"),
            ("c", "y"),
            ("c", "z"),
        ),
    )
    cover = factor_provider_cover(residual)
    cover.verify()
    assert cover.tie_break == COVER_TIE_BREAK
    assert cover.formal_concept_statistics.intent_subsets_examined == 8
    assert cover.formal_concept_statistics.maximal_biclique_candidates == len(
        cover.candidate_rectangles
    )
    assert all(
        rectangle.pairs <= frozenset(residual.pairs)
        for rectangle in cover.candidate_rectangles
    )
    assert frozenset(
        pair for rectangle in cover.rectangles for pair in rectangle.pairs
    ) == frozenset(residual.pairs)
    assert len(cover.rectangles) == 2


def test_provider_cover_counts_multiple_exact_minimum_ties() -> None:
    residual = relation(
        ("a", "b", "c"),
        ("x", "y", "z"),
        (
            ("a", "x"),
            ("a", "y"),
            ("b", "y"),
            ("b", "z"),
            ("c", "x"),
            ("c", "z"),
        ),
    )
    cover = factor_provider_cover(residual)
    assert len(cover.rectangles) == 3
    assert cover.optimal_cover_count >= 2
    assert cover.cover_statistics.combinations_examined > 0


def test_empty_residual_has_one_intent_one_empty_optimum_and_no_combinations() -> None:
    empty = relation(("zero",), ("unused",), ())
    cover = factor_provider_cover(empty)
    assert cover.rectangles == ()
    assert cover.candidate_rectangles == ()
    assert cover.formal_concept_statistics == FormalConceptSearchStatistics(1, 0)
    assert cover.cover_statistics == ProviderCoverSearchStatistics(0, 1)


def test_atomic_certificate_zero_degree_is_not_a_choke_point() -> None:
    required = relation(
        ("derived", "zero"),
        ("x",),
        (("derived", "x"),),
    )
    certificate = RelationFactorizationCertificate.factor(required)
    certificate.verify()
    surfaces = {
        surface.implementation_key: surface
        for surface in certificate.invalidation_surfaces
    }
    assert surfaces["derived"].is_choke_point
    assert not surfaces["zero"].is_choke_point
    assert surfaces["zero"].derived_consumers == ()
    with pytest.raises(ValueError):
        OwnerInvalidationSurface("zero", (), (), True)


def test_invalidation_surface_rejects_wrong_type_duplicate_and_foreign_rows() -> None:
    pair = RequiredRelationPair("owner", "x")
    with pytest.raises(TypeError):
        OwnerInvalidationSurface(
            "owner",
            (),
            ("not-a-pair",),
            False,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        OwnerInvalidationSurface("owner", (), (pair, pair), False)
    with pytest.raises(ValueError):
        OwnerInvalidationSurface(
            "owner", (), (RequiredRelationPair("foreign", "x"),), False
        )


def test_canonical_json_and_digest_ignore_input_iteration_order() -> None:
    first = RelationFactorizationCertificate.factor(crossing_relation())
    second = RelationFactorizationCertificate.factor(
        relation(
            ("zero", "right", "left"),
            ("c", "b", "a"),
            (("right", "c"), ("right", "b"), ("left", "b"), ("left", "a")),
        )
    )
    assert first.schema == CERTIFICATE_SCHEMA
    assert first.canonical_json() == second.canonical_json()
    assert first.content_digest() == second.content_digest()
    assert "retained" not in first.to_dict()["laminar_repair"]["arrangement"]


@pytest.mark.parametrize("value", [True, 1.5, "1", -1, 0])
def test_laminar_statistics_reject_nonexact_or_nonpositive_counts(
    value: object,
) -> None:
    for field_name in (
        "subsets_examined",
        "feasible_laminar_subsets",
        "optimal_subsets",
    ):
        values = {
            "subsets_examined": 1,
            "feasible_laminar_subsets": 1,
            "optimal_subsets": 1,
        }
        values[field_name] = value
        with pytest.raises((TypeError, ValueError)):
            LaminarSearchStatistics(**values)  # type: ignore[arg-type]


def test_laminar_statistics_reject_impossible_relations() -> None:
    with pytest.raises(ValueError):
        LaminarSearchStatistics(1, 2, 1)
    with pytest.raises(ValueError):
        LaminarSearchStatistics(2, 1, 2)


@pytest.mark.parametrize("value", [True, 1.5, "1", -1, 0])
def test_formal_statistics_reject_invalid_positive_intent_count(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        FormalConceptSearchStatistics(value, 0)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, 1.5, "1", -1])
def test_formal_statistics_reject_invalid_candidate_count(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        FormalConceptSearchStatistics(1, value)  # type: ignore[arg-type]


def test_formal_statistics_reject_impossible_relations() -> None:
    with pytest.raises(ValueError):
        FormalConceptSearchStatistics(1, 2)


@pytest.mark.parametrize("value", [True, 1.5, "1", -1])
def test_cover_statistics_reject_invalid_combination_count(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ProviderCoverSearchStatistics(value, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, 1.5, "1", -1, 0])
def test_cover_statistics_reject_invalid_optimal_count(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ProviderCoverSearchStatistics(1, value)  # type: ignore[arg-type]


def test_cover_statistics_reject_impossible_relations() -> None:
    with pytest.raises(ValueError):
        ProviderCoverSearchStatistics(0, 2)
    with pytest.raises(ValueError):
        ProviderCoverSearchStatistics(1, 2)


@pytest.mark.parametrize(
    "limits",
    [
        ExactSearchLimits(max_pairs=1),
        ExactSearchLimits(max_implementation_keys=1),
        ExactSearchLimits(max_consumers=1),
        ExactSearchLimits(max_laminar_subsets=8),
    ],
)
def test_relation_and_laminar_bounds_fail_loudly(limits: ExactSearchLimits) -> None:
    with pytest.raises(ExactSearchLimitExceeded):
        factor_laminar_relation(crossing_relation(), limits=limits)


def test_fca_candidate_and_cover_bounds_fail_loudly() -> None:
    residual = relation(
        ("a", "b", "c"),
        ("x", "y", "z"),
        (
            ("a", "x"),
            ("a", "y"),
            ("b", "y"),
            ("b", "z"),
            ("c", "x"),
            ("c", "z"),
        ),
    )
    with pytest.raises(ExactSearchLimitExceeded):
        factor_provider_cover(
            residual, limits=ExactSearchLimits(max_fca_intent_subsets=4)
        )
    with pytest.raises(ExactSearchLimitExceeded):
        factor_provider_cover(
            residual, limits=ExactSearchLimits(max_provider_candidates=1)
        )
    with pytest.raises(ExactSearchLimitExceeded):
        factor_provider_cover(
            residual, limits=ExactSearchLimits(max_cover_combinations=1)
        )


def test_search_limits_reject_nonpositive_and_boolean_values() -> None:
    with pytest.raises(ValueError):
        ExactSearchLimits(max_pairs=0)
    with pytest.raises(ValueError):
        ExactSearchLimits(max_pairs=True)


def test_laminar_certificate_rejects_relation_objective_gap_and_statistics_forgery() -> (
    None
):
    repair = factor_laminar_relation(crossing_relation())
    alternate = repair.required.with_pairs(repair.required.pairs[:2])
    for forged in (
        replace(repair, retained=alternate),
        replace(repair, residual=alternate),
        replace(repair, ancestry_gap=()),
        replace(repair, tie_break="forged"),
        replace(
            repair,
            statistics=replace(
                repair.statistics,
                subsets_examined=repair.statistics.subsets_examined + 1,
            ),
        ),
        replace(
            repair,
            statistics=replace(
                repair.statistics,
                feasible_laminar_subsets=(
                    repair.statistics.feasible_laminar_subsets + 1
                ),
            ),
        ),
        replace(
            repair,
            statistics=replace(
                repair.statistics,
                optimal_subsets=repair.statistics.optimal_subsets + 1,
            ),
        ),
    ):
        with pytest.raises(ValueError):
            forged.verify()


def test_laminar_certificate_rejects_forged_arrangement_maps() -> None:
    repair = factor_laminar_relation(crossing_relation())
    arrangement = repair.arrangement
    with pytest.raises(ValueError):
        SingleParentArrangement(
            implementation_parents=tuple(reversed(arrangement.implementation_parents)),
            consumer_parents=arrangement.consumer_parents,
            tie_break=arrangement.tie_break,
            retained=repair.retained,
        )
    with pytest.raises(ValueError):
        SingleParentArrangement(
            implementation_parents=arrangement.implementation_parents,
            consumer_parents=tuple(reversed(arrangement.consumer_parents)),
            tie_break=arrangement.tie_break,
            retained=repair.retained,
        )


def test_provider_cover_rejects_rectangle_candidate_union_count_and_policy_forgery() -> (
    None
):
    cover = factor_provider_cover(crossing_relation())
    fake = ProviderRectangle(("left",), ("a",))
    for forged in (
        replace(cover, rectangles=(fake,)),
        replace(cover, candidate_rectangles=()),
        replace(cover, tie_break="forged"),
        replace(
            cover,
            formal_concept_statistics=replace(
                cover.formal_concept_statistics,
                intent_subsets_examined=(
                    cover.formal_concept_statistics.intent_subsets_examined + 1
                ),
            ),
        ),
        replace(
            cover,
            formal_concept_statistics=replace(
                cover.formal_concept_statistics,
                maximal_biclique_candidates=(
                    cover.formal_concept_statistics.maximal_biclique_candidates + 1
                ),
            ),
        ),
        replace(
            cover,
            cover_statistics=replace(
                cover.cover_statistics,
                combinations_examined=cover.cover_statistics.combinations_examined + 1,
            ),
        ),
        replace(
            cover,
            cover_statistics=replace(
                cover.cover_statistics,
                optimal_covers=cover.cover_statistics.optimal_covers + 1,
            ),
        ),
    ):
        with pytest.raises(ValueError):
            forged.verify()


def test_atomic_certificate_rejects_invalidation_and_nested_forgery() -> None:
    certificate = RelationFactorizationCertificate.factor(crossing_relation())
    surface_index = next(
        index
        for index, surface in enumerate(certificate.invalidation_surfaces)
        if surface.residual_pairs
    )
    surface = certificate.invalidation_surfaces[surface_index]
    forged_surface = replace(
        surface,
        residual_pairs=(),
        is_choke_point=bool(surface.derived_consumers),
    )
    forged = replace(
        certificate,
        invalidation_surfaces=(
            *certificate.invalidation_surfaces[:surface_index],
            forged_surface,
            *certificate.invalidation_surfaces[surface_index + 1 :],
        ),
    )
    with pytest.raises(ValueError):
        forged.verify()
    forged_cover = replace(
        certificate.provider_cover,
        cover_statistics=replace(
            certificate.provider_cover.cover_statistics,
            combinations_examined=(
                certificate.provider_cover.cover_statistics.combinations_examined + 1
            ),
        ),
    )
    with pytest.raises(ValueError):
        replace(certificate, provider_cover=forged_cover).verify()
    with pytest.raises(ValueError):
        replace(certificate, schema="other").verify()


def test_one_field_relation_mutation_changes_digest_and_old_certificate_fails() -> None:
    certificate = RelationFactorizationCertificate.factor(crossing_relation())
    mutated_relation = certificate.relation.with_pairs(certificate.relation.pairs[:-1])
    mutated = RelationFactorizationCertificate.factor(mutated_relation)
    assert certificate.content_digest() != mutated.content_digest()
    with pytest.raises(ValueError):
        replace(certificate, relation=mutated_relation).verify()


def test_typed_statistics_are_the_only_count_authorities() -> None:
    certificate = RelationFactorizationCertificate.factor(crossing_relation())
    repair_fields = set(certificate.laminar_repair.__dataclass_fields__)
    cover_fields = set(certificate.provider_cover.__dataclass_fields__)
    assert "optimal_repair_count" not in repair_fields
    assert "optimal_cover_count" not in cover_fields
    assert set(LaminarSearchStatistics.__dataclass_fields__) == {
        "subsets_examined",
        "feasible_laminar_subsets",
        "optimal_subsets",
    }
