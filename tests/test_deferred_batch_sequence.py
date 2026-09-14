"""Demanded collection access preserves original objects and complete tuple behavior."""

import pickle

import pytest

from nominal_refactor_advisor.collection_algebra import DeferredBatchSequence


class Batches(DeferredBatchSequence):
    def __init__(self, batches):
        super().__init__()
        self.batches = batches
        self.loads = []
        self.fail = set()

    @property
    def batch_count(self):
        return len(self.batches)

    def _load_batch(self, index):
        self.loads.append(index)
        if index in self.fail:
            raise ValueError("unavailable batch")
        return self.batches[index]


def test_partial_iteration_reads_only_the_required_original_batches():
    first, last = object(), object()
    values = Batches(((first,), (), (last,)))
    iterator = iter(values)
    assert values.loads == []
    assert next(iterator) is first
    assert values.loads == [0]
    assert next(iter(values)) is first
    assert values.loads == [0]
    assert next(iterator) is last
    assert values.loads == [0, 1, 2]
    assert tuple(values) == (first, last)
    assert values.materialized_item_count == 2
    assert values.loads == [0, 1, 2]


@pytest.mark.parametrize("index", (0, 2, -1, slice(None), slice(None, None, -1)))
def test_indexing_has_complete_tuple_semantics(index):
    values = Batches(((1, 2), (), (3,)))
    assert values[index] == (1, 2, 3)[index]
    assert values.loads == [0, 1, 2]
    assert len(values) == 3
    assert values.loads == [0, 1, 2]


def test_equality_hash_and_repr_use_values_not_loader_identity():
    left = Batches(((1,), (2, 3)))
    right = Batches(((1, 2), (3,)))
    assert left == right == (1, 2, 3)
    assert (1, 2, 3) == left
    assert left != (1, 2)
    assert left != [1, 2, 3]
    assert hash(left) == hash((1, 2, 3))
    assert repr(left) == repr((1, 2, 3))


def test_failed_read_is_not_cached_as_an_empty_or_partial_tail():
    values = Batches(((1,), (2,)))
    values.fail.add(1)
    assert next(iter(values)) == 1
    with pytest.raises(ValueError, match="unavailable"):
        tuple(values)
    assert values.materialized_item_count == 1
    values.fail.clear()
    assert tuple(values) == (1, 2)
    assert values.loads == [0, 1, 1]


def test_recursive_load_rejects_without_poisoning_the_next_read():
    class Recursive(Batches):
        recursive = True

        def _load_batch(self, index):
            if self.recursive:
                return self._batch(index)
            return super()._load_batch(index)

    values = Recursive(((1,),))
    with pytest.raises(ValueError, match="Cyclic"):
        tuple(values)
    assert values.materialized_item_count == 0
    values.recursive = False
    assert tuple(values) == (1,)


def test_pickle_materializes_values_and_preserves_shared_object_relations():
    shared = []
    values = Batches(((shared,), (shared,)))
    sequence, external = pickle.loads(pickle.dumps((values, shared)))
    assert type(sequence) is tuple
    assert sequence[0] is sequence[1] is external
    assert values.loads == [0, 1]


def test_empty_sequence_has_no_fabricated_loads():
    values = Batches(())
    assert len(values) == 0
    assert tuple(values) == ()
    assert values.materialized_item_count == 0
    assert values.loads == []
