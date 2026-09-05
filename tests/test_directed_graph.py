"""Ordered reachability is a relation query, independent of class/MRO semantics."""

from enum import Enum, auto

from nominal_refactor_advisor.semantic_algebra import DirectedGraph


class Vertex(Enum):
    ROOT = auto()
    LEFT = auto()
    RIGHT = auto()
    LEAF = auto()


def test_graph_uses_nominal_vertices_and_caches_only_requested_roots() -> None:
    graph = DirectedGraph(
        {
            Vertex.ROOT: (Vertex.LEFT, Vertex.RIGHT),
            Vertex.LEFT: (Vertex.LEAF,),
            Vertex.RIGHT: (Vertex.LEAF,),
            Vertex.LEAF: (),
        }
    )
    assert "_reachability_by_vertex" not in vars(graph)
    result = graph.reachable_from(Vertex.ROOT)
    assert result == (Vertex.LEFT, Vertex.RIGHT, Vertex.LEAF)
    assert graph.reachable_from(Vertex.ROOT) is result
    assert tuple(graph._reachability_by_vertex) == (Vertex.ROOT,)
    assert graph.reversed.reachable_from(Vertex.LEAF) == (
        Vertex.LEFT,
        Vertex.RIGHT,
        Vertex.ROOT,
    )
    assert graph.reversed is graph.reversed


def test_cycles_self_edges_and_duplicate_edges_terminate_in_breadth_first_order() -> (
    None
):
    graph = DirectedGraph({"a": ("b", "b"), "b": ("c", "a"), "c": ("c",)})
    assert graph.reachable_from("a") == ("b", "c", "a")
    assert graph.reachable_from("b") == ("c", "a", "b")
    assert graph.reachable_from("c") == ("c",)
    assert graph.reversed.neighbors == {"a": ("b",), "b": ("a", "a"), "c": ("b", "c")}


def test_dangling_vertex_and_unknown_root_preserve_relation_boundaries() -> None:
    graph = DirectedGraph({1: (2,), 3: ()})
    assert graph.reachable_from(1) == (2,)
    assert graph.reachable_from(2) == graph.reachable_from(4) == ()
    assert graph.nonempty_reachability_from((1, 2, 3, 4)) == {1: (2,)}
    assert graph.reversed.reachable_from(2) == (1,)
    assert graph.reversed.nonempty_reachability_from(graph.neighbors) == {}


def test_large_breadth_retains_declaration_order() -> None:
    leaves = tuple(range(1, 10001))
    graph = DirectedGraph({0: leaves})
    assert graph.reachable_from(0) == leaves
