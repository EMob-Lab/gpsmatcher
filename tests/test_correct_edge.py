import networkx as nx
import numpy as np

from gpsmatcher.map_matching import correct_edge


def test_neighbour_edge_outside_the_candidates():
    # road 1 -> 2 -> 3 -> 4; the GPS points are near (1, 2) and (3, 4) only, so (2, 3), the successor of (1, 2)
    # and the predecessor of (3, 4) on the path, is not a candidate edge of the trace
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
    id2edges = {0: (1, 2), 1: (2, 3), 2: (3, 4)}
    edges2id = {edge: i for i, edge in id2edges.items()}
    sub_edges = [0, 2]
    states = np.arange(len(sub_edges))
    state = np.array([0., 1.])                  # Viterbi path: (1, 2) then (3, 4)
    emit_p = np.array([[3.9, 0.], [0., 3.9]])   # one point on each edge
    new_state = correct_edge(graph, states, state, sub_edges, id2edges, edges2id, emit_p)
    assert [id2edges[sub_edges[int(i)]] for i in new_state] == [(1, 2), (3, 4)]


def test_neighbour_edge_with_a_better_emission():
    # the second point is closer to (2, 3), a candidate edge, than to (3, 4) where Viterbi put it
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
    id2edges = {0: (1, 2), 1: (2, 3), 2: (3, 4)}
    edges2id = {edge: i for i, edge in id2edges.items()}
    sub_edges = [0, 1, 2]
    states = np.arange(len(sub_edges))
    state = np.array([0., 2.])
    emit_p = np.array([[3.9, 0., 0.], [0., 3.5, 1.0]])
    new_state = correct_edge(graph, states, state, sub_edges, id2edges, edges2id, emit_p)
    assert [id2edges[sub_edges[int(i)]] for i in new_state] == [(1, 2), (2, 3)]
