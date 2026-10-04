"""
The current code against gpsmatcher 0.1.1 (tests/reference_0_1_1.py) on generated graphs and traces
(tests/generate.py): the outputs must be identical, cell by cell.
"""
import numpy as np
import pandas as pd
import pytest

import reference_0_1_1 as reference
from generate import make_case
from gpsmatcher.map_matching import mm_gps

SEEDS = range(300)


def _same_cell(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, list) or isinstance(b, list):
        return type(a) is type(b) and len(a) == len(b) and all(_same_cell(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return type(a) is type(b) and a == b


def assert_same_output(old: pd.DataFrame, new: pd.DataFrame):
    pd.testing.assert_index_equal(old.index, new.index, exact=True)
    assert list(old.columns) == list(new.columns)
    assert list(old.dtypes) == list(new.dtypes)
    for column in old.columns:
        for i, (a, b) in enumerate(zip(old[column], new[column])):
            assert _same_cell(a, b), f"column {column}, row {i}: {a!r} != {b!r}"


def run(mm, case):
    """mm_gps of a case on copies of its inputs (mm_gps changes its gps frame); the exception type if it raises."""
    try:
        return mm(case.gps.copy(), case.G_mm, case.trans, case.candidates, case.id2edges, case.edges2id,
                  alpha=case.alpha, radius=case.radius)
    except Exception as e:
        return type(e)


def reference_reads_past_emission(case) -> set:
    """Trips whose emission rows go beyond the emission matrix: 0.1.1 reads past the end of an array for them."""
    gps, gps_mm, dic_geohash, dic_candidates = reference.process_gps(case.gps.copy(), case.candidates)
    emit = reference.emission_matrix(gps, case.G_mm, dic_geohash, dic_candidates, alpha=case.alpha,
                                     radius=case.radius, show_print=False)
    return {trip for trip, row in gps_mm.iterrows() if row['first_emission'] + len(row['traj']) > emit.shape[0]}


@pytest.mark.parametrize('seed', SEEDS)
def test_mm_gps_identical(seed):
    case = make_case(seed)
    old, new = run(reference.mm_gps, case), run(mm_gps, case)
    if isinstance(old, type):
        assert new is old
        return
    assert not isinstance(new, type), f"the new code raised {new}"
    _, old_gps, old_mm = old
    _, new_gps, new_mm = new
    past = reference_reads_past_emission(case)
    if past:
        # 0.1.1's result depends on the memory read past the array; the new code reads the missing rows as points
        # without emission, so the trace has probability 0: NaN
        for trip in past:
            edges = new_mm.at[trip, 'edges']
            assert len(edges) == len(new_mm.at[trip, 'traj']) and all(np.isnan(e) for e in edges)
            assert new_mm.at[trip, 'shortest_path_nodes'] == []
            old_mm.at[trip, 'map_match'] = new_mm.at[trip, 'map_match']
            old_mm.at[trip, 'edges'] = new_mm.at[trip, 'edges']
            old_mm.at[trip, 'shortest_path_nodes'] = new_mm.at[trip, 'shortest_path_nodes']
        return assert_same_output(old_mm, new_mm)
    assert_same_output(old_gps, new_gps)
    assert_same_output(old_mm, new_mm)


def _trip_facts(case) -> list:
    """Per trip of the reference run: points, states, a tie of best emissions, prob == 0, inf in V, outcome."""
    try:
        gps, gps_mm, dic_geohash, dic_candidates = reference.process_gps(case.gps.copy(), case.candidates)
        emit = reference.emission_matrix(gps, case.G_mm, dic_geohash, dic_candidates, alpha=case.alpha,
                                         radius=case.radius, show_print=False)
    except Exception:
        return [{'empty_emission': True}]
    facts = []
    for trip, row in gps_mm.iterrows():
        start, n, sub_edges = row['first_emission'], len(row['traj']), row['sub_edges']
        fact = {'points': n, 'states': len(sub_edges), 'empty_emission': False}
        if start + n <= emit.shape[0] and max(sub_edges) < min(case.trans.shape):
            emit_p = emit[start:start + n, :][:, sub_edges].toarray() / 100
            trans_p = case.trans[sub_edges, :][:, sub_edges].toarray() / 100
            V, prob, _, _ = reference.fast_viterbi(np.arange(n), np.arange(len(sub_edges)),
                                                   np.ones(len(sub_edges)) / len(sub_edges), trans_p, emit_p)
            top = np.sort(emit_p, axis=1)[:, -2:] if emit_p.shape[1] > 1 else None
            fact.update(tie=top is not None and bool(((top[:, 0] == top[:, 1]) & (top[:, 1] > 0)).any()),
                        prob_zero=prob == 0, inf=bool(np.isinf(V).any()))
        edges = reference.one_traj_mm(row['traj'], case.G_mm, case.trans, emit, case.id2edges, case.edges2id,
                                      sub_edges, start, row['last_emission'])[0]
        fact['outcome'] = ('problem' if edges and edges[0] == "Problem" else
                           'nan' if edges and isinstance(edges[0], float) else 'matched')
        facts.append(fact)
    return facts


def test_generator_covers_edge_cases():
    kinds, facts = {}, []
    for seed in SEEDS:
        case = make_case(seed)
        for kind in case.kinds:
            kinds[kind] = kinds.get(kind, 0) + 1
        facts += _trip_facts(case)
    trips = [f for f in facts if not f['empty_emission']]
    counts = {
        'tie': sum(f.get('tie', False) for f in trips),
        'single_state': sum(f['states'] == 1 for f in trips),
        'short_nan': sum(f['outcome'] == 'nan' and f['points'] < 100 for f in trips),
        'problem': sum(f['outcome'] == 'problem' for f in trips),
        'empty_emission': sum(f['empty_emission'] for f in facts),
        'long_prob_zero': sum(f.get('prob_zero', False) and f['points'] >= 500 for f in trips),
        'inf': sum(f.get('inf', False) for f in trips),
    }
    print(kinds, counts)
    for kind in ('far_start', 'far_middle', 'far_end', 'all_far', 'single_point', 'stationary', 'on_node',
                 'long_noisy', 'long_stationary', 'non_contiguous', 'foreign_transition'):
        assert kinds.get(kind, 0) >= 5, kind
    assert counts['tie'] >= 100
    assert counts['single_state'] >= 5
    assert counts['short_nan'] >= 5
    assert counts['problem'] >= 3
    assert counts['empty_emission'] >= 3
    assert counts['long_prob_zero'] >= 1
    assert counts['inf'] >= 1
