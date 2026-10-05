import numba
import numpy as np


def _jit(function):
    """numba.njit with its cache on disk, or without it where numba finds no cache directory it can write."""
    try:
        return numba.njit(cache=True)(function)
    except RuntimeError:
        return numba.njit(function)


def sparse_viterbi(n_obs, start, sub_edges, edge2state, emit_indptr, emit_indices, emit_data,
                   trans_indptr, trans_indices, trans_data):
    """
    Viterbi algorithm of a trace, reading the emission and transition matrices in CSR form: same result as
    fast_viterbi on the dense matrices that one_traj_mm extracted with scipy in version 0.1.1, without building them.

    Parameters
    ----------
    n_obs : int
        Number of GPS points of the trace.

    start : int
        Row of the first point of the trace in the emission matrix (the trace has rows start to start + n_obs - 1).
        Rows beyond the matrix are points without emission: the matrix ends at the last point that has one.

    sub_edges : numpy.ndarray
        Candidate edges of the trace (int64): state s is edge sub_edges[s].

    edge2state : numpy.ndarray
        Work array (int64) of at least as many elements as the matrices have columns, all -1; restored on return.

    emit_indptr, emit_indices, emit_data : numpy.ndarray
        Emission matrix (points x edges) in CSR form.

    trans_indptr, trans_indices, trans_data : numpy.ndarray
        Transition matrix (edges x edges) in CSR form.

    Returns
    -------
    prob : float
        Probability of the most likely path.

    seq : numpy.ndarray
        State of each point on the most likely path (zeros if prob is 0).

    emit_p : numpy.ndarray
        Emission probabilities of the trace (points x states), as correct_edge reads them.
    """
    n_states = sub_edges.shape[0]
    for s in range(n_states):
        edge2state[sub_edges[s]] = s

    # emissions of the trace: values summed in the dtype of the matrix, as scipy does for duplicate entries, then
    # divided by 100 as 0.1.1 divided its dense extraction
    counts = np.zeros((n_obs, n_states), dtype=emit_data.dtype)
    n_rows = emit_indptr.shape[0] - 1
    for t in range(min(n_obs, n_rows - start)):
        for k in range(emit_indptr[start + t], emit_indptr[start + t + 1]):
            s = edge2state[emit_indices[k]]
            if s >= 0:
                counts[t, s] += emit_data[k]
    emit_p = np.empty((n_obs, n_states))
    for t in range(n_obs):
        for s in range(n_states):
            emit_p[t, s] = counts[t, s] / 100

    # rows of the transitions between the states of the trace, read when a state is first a predecessor
    trans_p = np.zeros((n_states, n_states))
    loaded = np.zeros(n_states, dtype=np.bool_)
    row_counts = np.zeros(n_states, dtype=trans_data.dtype)

    v_prev = np.empty(n_states)
    start_p = 1.0 / n_states
    for y in range(n_states):
        v_prev[y] = start_p * emit_p[0, y]
    # state of the previous point on the best path to each state, instead of 0.1.1's matrix of whole paths
    back = np.zeros((n_obs, n_states), dtype=np.int64)
    prev = np.empty(n_states, dtype=np.int64)
    for t in range(1, n_obs):
        n_prev = 0
        for y0 in range(n_states):
            if v_prev[y0] != 0:
                prev[n_prev] = y0
                n_prev += 1
                if not loaded[y0]:
                    row_counts[:] = 0
                    edge = sub_edges[y0]
                    for k in range(trans_indptr[edge], trans_indptr[edge + 1]):
                        s = edge2state[trans_indices[k]]
                        if s >= 0:
                            row_counts[s] += trans_data[k]
                    for s in range(n_states):
                        trans_p[y0, s] = row_counts[s] / 100
                    loaded[y0] = True
        v = np.zeros(n_states)
        for y in range(n_states):
            if emit_p[t, y] != 0:
                prob = 0.0
                best = 0
                # predecessors in increasing order, strict >: the first state wins a tie
                for i in range(n_prev):
                    y0 = prev[i]
                    if v_prev[y0] * trans_p[y0, y] * emit_p[t, y] > prob:
                        prob = v_prev[y0] * trans_p[y0, y] * emit_p[t, y]
                        best = y0
                v[y] = prob
                if prob > 0:
                    back[t, y] = best
        v_prev = v

    state = np.argmax(v_prev)
    prob = v_prev[state]
    seq = np.zeros(n_obs)
    if prob != 0:
        seq[n_obs - 1] = state
        for t in range(n_obs - 1, 0, -1):
            state = back[t, state]
            seq[t - 1] = state

    for s in range(n_states):
        edge2state[sub_edges[s]] = -1
    return prob, seq, emit_p


sparse_viterbi = _jit(sparse_viterbi)
