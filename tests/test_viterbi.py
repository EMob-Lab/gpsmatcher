"""The Viterbi on the sparse matrices (gpsmatcher.viterbi) against 0.1.1's dense one (tests/reference_0_1_1.py)."""
import networkx as nx
import numpy as np
import pytest
from scipy.sparse import csr_matrix

import reference_0_1_1 as reference
from generate import make_kernel_case
from gpsmatcher.map_matching import one_traj_mm
from gpsmatcher.viterbi import sparse_viterbi


def _sparse(emission, trans, sub_edges, start, n_obs):
    edge2state = np.full(emission.shape[1], -1, dtype=np.int64)
    out = sparse_viterbi(n_obs, start, np.asarray(sub_edges, dtype=np.int64), edge2state, emission.indptr,
                         emission.indices, emission.data, trans.indptr, trans.indices, trans.data)
    return out, edge2state


def _dense(emission, trans, sub_edges, start, n_obs):
    """0.1.1's one_traj_mm up to the Viterbi."""
    emit_p = emission[start:start + n_obs, :][:, sub_edges].toarray() / 100
    trans_p = trans[sub_edges, :][:, sub_edges].toarray() / 100
    start_p = np.ones(len(sub_edges)) / len(sub_edges)
    _, prob, seq, _ = reference.fast_viterbi(np.arange(n_obs), np.arange(len(sub_edges)), start_p, trans_p, emit_p)
    return prob, seq, emit_p


@pytest.mark.parametrize('seed', range(3000))
def test_kernel_identical(seed):
    case = make_kernel_case(seed)
    old_prob, old_seq, old_emit = _dense(case.emission, case.trans, case.sub_edges, case.start, case.n_obs)
    (prob, seq, emit_p), _ = _sparse(case.emission, case.trans, case.sub_edges, case.start, case.n_obs)
    assert prob == old_prob or (np.isinf(prob) and np.isinf(old_prob))
    assert np.array_equal(emit_p, old_emit)
    if prob != 0:
        assert np.array_equal(seq, old_seq)


def _matrices(emit_rows, trans_rows):
    return csr_matrix(np.array(emit_rows, dtype=np.int16)), csr_matrix(np.array(trans_rows, dtype=np.int8))


def test_single_point():
    emission, trans = _matrices([[300, 100]], [[100, 50], [50, 100]])
    (prob, seq, _), _ = _sparse(emission, trans, [0, 1], 0, 1)
    assert prob == 1.5
    assert np.array_equal(seq, [0.])


def test_tie_keeps_first_state():
    emission, trans = _matrices([[200, 200]] * 4, [[100, 100], [100, 100]])
    (prob, seq, _), _ = _sparse(emission, trans, [0, 1], 0, 4)
    assert prob > 0
    assert np.array_equal(seq, [0., 0., 0., 0.])


def test_duplicate_entries_are_summed():
    # row 0 of the transitions stores column 1 twice: 50 + 30
    trans = csr_matrix((np.array([50, 30, 100], dtype=np.int8), np.array([1, 1, 1]), np.array([0, 2, 3])),
                       shape=(2, 2))
    emission = csr_matrix(np.array([[100, 0], [0, 100]], dtype=np.int16))
    (prob, seq, _), _ = _sparse(emission, trans, [0, 1], 0, 2)
    assert prob == 0.5 * 1.0 * 0.8 * 1.0
    assert np.array_equal(seq, [0., 1.])


def test_edge2state_restored():
    case = make_kernel_case(7)
    _, edge2state = _sparse(case.emission, case.trans, case.sub_edges, case.start, case.n_obs)
    assert (edge2state == -1).all()


def _line_graph():
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
    id2edges = {0: (1, 2), 1: (2, 3), 2: (3, 4)}
    return graph, id2edges, {edge: i for i, edge in id2edges.items()}


def test_trailing_points_beyond_emission():
    # 3 points, but the emission matrix has rows for the first 2 only (the last point has no edge within the radius)
    graph, id2edges, edges2id = _line_graph()
    trans = csr_matrix(np.array([[100, 80, 50], [0, 100, 80], [0, 0, 100]], dtype=np.int8))
    emission = csr_matrix((np.array([300, 200], dtype=np.int16), (np.array([0, 1]), np.array([0, 1]))), shape=(2, 3))
    edges, path = one_traj_mm(np.zeros((3, 2)), graph, trans, emission, id2edges, edges2id, [0, 1, 2], 0, 2)
    assert len(edges) == 3 and all(np.isnan(e) for e in edges) and path == []


def test_candidate_beyond_transition_matrix():
    # gpsmatcher's transition_matrix has max(edge with a transition) + 1 rows: scipy's extraction raised IndexError
    graph, id2edges, edges2id = _line_graph()
    trans = csr_matrix(np.array([[100, 80], [0, 100]], dtype=np.int8))
    emission = csr_matrix(np.array([[300, 0, 0], [0, 200, 100]], dtype=np.int16))
    args = (np.zeros((2, 2)), graph, trans, emission, id2edges, edges2id, [0, 1, 2], 0, 1)
    assert reference.one_traj_mm(*args) == (["Problem"] * 2, ["Problem"])
    assert one_traj_mm(*args) == (["Problem"] * 2, ["Problem"])
