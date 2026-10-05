"""
Seeded generator of map matching inputs for the identity tests: small road graphs prepared by gpsmatcher itself, GPS
traces on them with the edge cases the Viterbi, correct_edge and the path assembly must handle, and random sparse
matrices for the Viterbi alone.
"""
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd
import pygeohash_fast
from scipy.sparse import csr_matrix
from shapely.geometry import LineString, Point

from gpsmatcher.graph import process_graph
from gpsmatcher.precomputation_emission import process_dic_cand_edges
from gpsmatcher.transition import transition_matrix

LON0, LAT0 = 2.35, 48.85
M_PER_DEG_LON, M_PER_DEG_LAT = 111_320 * np.cos(np.radians(LAT0)), 110_540
FAR_M = 5_000      # far enough from every edge of a grid to have no candidate edge at all

TRIP_KINDS = {'far_start': 0.05, 'far_middle': 0.05, 'far_end': 0.05, 'all_far': 0.02, 'single_point': 0.05,
              'stationary': 0.05, 'on_node': 0.05, 'long_noisy': 0.03, 'long_stationary': 0.03}


@dataclass
class Case:
    G_mm: nx.DiGraph
    trans: csr_matrix
    candidates: dict
    id2edges: dict
    edges2id: dict
    gps: pd.DataFrame
    alpha: float
    radius: float
    kinds: set = field(default_factory=set)


@dataclass
class KernelCase:
    emission: csr_matrix
    trans: csr_matrix
    sub_edges: list
    start: int
    n_obs: int


def _lonlat(x, y):
    return LON0 + np.asarray(x) / M_PER_DEG_LON, LAT0 + np.asarray(y) / M_PER_DEG_LAT


def _edge(rng, p, q, bent):
    (x0, y0), (x1, y1) = p, q
    coords = [p, q]
    if bent:
        length = np.hypot(x1 - x0, y1 - y0)
        offset = rng.uniform(-0.3, 0.3) * length
        mx, my = (x0 + x1) / 2 - offset * (y1 - y0) / length, (y0 + y1) / 2 + offset * (x1 - x0) / length
        coords = [p, (mx, my), q]
    lon, lat = _lonlat([c[0] for c in coords], [c[1] for c in coords])
    line_m = LineString(coords)
    return LineString(zip(lon, lat)), line_m.length


def _graph(rng, foreign):
    nx_, ny_ = rng.integers(3, 9, size=2)
    spacing = rng.uniform(60, 400)
    xy = {i * ny_ + j + 1: (i * spacing + rng.uniform(-10, 10), j * spacing + rng.uniform(-10, 10))
          for i in range(nx_) for j in range(ny_)}
    links = [(i * ny_ + j + 1, (i + 1) * ny_ + j + 1) for i in range(nx_ - 1) for j in range(ny_)]
    links += [(i * ny_ + j + 1, i * ny_ + j + 2) for i in range(nx_) for j in range(ny_ - 1)]
    G = nx.DiGraph()
    for u, v in links:
        if rng.random() > 0.8:
            continue
        both = rng.random() < 0.7
        if not both and rng.random() < 0.5:
            u, v = v, u
        geometry, length = _edge(rng, xy[u], xy[v], rng.random() < 0.3)
        speed = rng.choice([5, 8, 14, 25])
        G.add_edge(u, v, geometry=geometry, weight=length / speed)
        if both:
            G.add_edge(v, u, geometry=LineString(list(geometry.coords)[::-1]), weight=length / speed)
    if foreign:
        # a one-way segment beside the grid, not connected to it
        a, b = max(xy) + 1, max(xy) + 2
        x = nx_ * spacing + 300
        xy[a], xy[b] = (x, 0.0), (x, spacing)
        geometry, length = _edge(rng, xy[a], xy[b], False)
        G.add_edge(a, b, geometry=geometry, weight=length / 14)
    return G, xy


def _walk(rng, G_mm, length_m):
    """Points every few metres along a random walk on the graph, in lon/lat, without noise."""
    edges = list(G_mm.edges)
    u, v = edges[rng.integers(len(edges))]
    step = rng.uniform(5, 80)
    lon, lat, walked = [], [], 0.0
    while walked < length_m:
        line = G_mm.edges[u, v]['geometry']
        for f in np.arange(rng.uniform(0, 1) * 0.2, 1, 0.2):
            point = line.interpolate(f, normalized=True)
            lon.append(point.x)
            lat.append(point.y)
        walked += step * 5
        successors = list(G_mm.successors(v))
        if not successors:
            break
        u, v = v, successors[rng.integers(len(successors))]
    return np.array(lon), np.array(lat)


def _noise(rng, lon, lat, sigma_m):
    return (lon + rng.normal(0, sigma_m, len(lon)) / M_PER_DEG_LON,
            lat + rng.normal(0, sigma_m, len(lat)) / M_PER_DEG_LAT)


def _trip(rng, G_mm, xy, kinds):
    if 'long_stationary' in kinds:
        line = G_mm.edges[list(G_mm.edges)[rng.integers(G_mm.number_of_edges())]]['geometry']
        point = line.interpolate(0.5, normalized=True)
        return np.full(700, point.x), np.full(700, point.y)
    if 'long_noisy' in kinds:
        lon, lat = np.array([]), np.array([])
        while len(lon) < 1500:
            more = _walk(rng, G_mm, 5_000)
            lon, lat = np.r_[lon, more[0]], np.r_[lat, more[1]]
        return _noise(rng, lon[:1500], lat[:1500], 30)
    lon, lat = _walk(rng, G_mm, rng.uniform(50, 1500))
    n = int(rng.integers(2, 61))
    if len(lon) < 2:
        lon, lat = np.r_[lon, lon], np.r_[lat, lat]
    lon, lat = lon[:n], lat[:n]
    lon, lat = _noise(rng, lon, lat, rng.uniform(0, 40))
    if 'stationary' in kinds:
        k = int(rng.integers(len(lon)))
        repeat = int(rng.integers(2, 10))
        lon, lat = np.insert(lon, k, [lon[k]] * repeat), np.insert(lat, k, [lat[k]] * repeat)
    if 'on_node' in kinds:
        node = list(G_mm.nodes)[rng.integers(G_mm.number_of_nodes())]
        nlon, nlat = _lonlat(*xy[node])
        k = int(rng.integers(len(lon)))
        lon[k], lat[k] = nlon, nlat
    far = FAR_M / M_PER_DEG_LON
    for kind, k in (('far_start', 0), ('far_middle', len(lon) // 2), ('far_end', len(lon) - 1)):
        if kind in kinds:
            lon[k] += far
    if 'all_far' in kinds:
        lon = lon + far
    if 'single_point' in kinds:
        lon, lat = lon[:1], lat[:1]
    return lon, lat


def _foreign_trip(G_mm, xy):
    """Points along a grid edge e, then along the foreign segment f, which only the added transition e -> f joins."""
    f = max(xy) - 1, max(xy)
    e = next(edge for edge in G_mm.edges if edge != f)
    points = [G_mm.edges[edge]['geometry'].interpolate(t, normalized=True) for edge in (e, f) for t in (0.3, 0.7)]
    return np.array([p.x for p in points]), np.array([p.y for p in points]), e, f


def _add_transitions(trans, n, pairs):
    coo = trans.tocoo()
    rows = np.r_[coo.row, [p[0] for p in pairs]]
    cols = np.r_[coo.col, [p[1] for p in pairs]]
    data = np.r_[coo.data, np.full(len(pairs), 100, dtype=np.int8)].astype(np.int8)
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def _beyond_radius(rng, G_mm, candidates, radius):
    """Up to 6 points that have candidate edges (by geohash) but none within the radius: an empty emission matrix."""
    lines = {d['edge_id']: LineString([((x - LON0) * M_PER_DEG_LON, (y - LAT0) * M_PER_DEG_LAT)
                                       for x, y in d['geometry'].coords]) for _, _, d in G_mm.edges(data=True)}
    lon, lat = [], []
    for _ in range(400):
        line = lines[list(lines)[rng.integers(len(lines))]]
        point = line.interpolate(rng.uniform(0, 1), normalized=True)
        angle = rng.uniform(0, 2 * np.pi)
        d = radius + rng.uniform(5, 40)
        x, y = point.x + d * np.cos(angle), point.y + d * np.sin(angle)
        plon, plat = _lonlat(x, y)
        cands = candidates.get(geohashes([plon], [plat])[0])
        if cands and min(lines[e].distance(Point(x, y)) for e in cands) >= radius + 3 and \
                min(l.distance(Point(x, y)) for l in lines.values()) >= radius + 3:
            lon.append(float(plon))
            lat.append(float(plat))
            if len(lon) == 6:
                break
    return np.array(lon), np.array(lat)


def make_case(seed: int) -> Case:
    rng = np.random.default_rng(seed)
    foreign = rng.random() < 0.1
    G, xy = _graph(rng, foreign)
    G_mm, id2edges, edges2id = process_graph(G, max_length=np.inf, save=False, show_print=False)
    beta = rng.choice([1 / 36, 1 / 500])
    radius = float(rng.choice([50, 150]))
    alpha = float(rng.choice([0.02, 0.1]))
    trans = transition_matrix(G_mm, 120, beta=beta, save=False, show_print=False)
    candidates = process_dic_cand_edges(G_mm, radius=radius, save=False, show_print=False)

    case_kinds, trips = set(), []
    ids = rng.permutation(1000)[:int(rng.integers(1, 13))]
    for trip_id in ids:
        kinds = {kind for kind, p in TRIP_KINDS.items() if rng.random() < p}
        case_kinds |= kinds
        lon, lat = _trip(rng, G_mm, xy, kinds)
        trips.append(pd.DataFrame({'lon': lon, 'lat': lat, 'ID_trip': trip_id}))
    if foreign:
        lon, lat, e, f = _foreign_trip(G_mm, xy)
        trips.append(pd.DataFrame({'lon': lon, 'lat': lat, 'ID_trip': 1000 + seed % 1000}))
        trans = _add_transitions(trans, G_mm.number_of_edges(), [(edges2id[e], edges2id[f])])
        case_kinds.add('foreign_transition')
    if rng.random() < 0.03:
        lon, lat = _beyond_radius(rng, G_mm, candidates, radius)
        if len(lon) >= 2:
            trips = [pd.DataFrame({'lon': lon, 'lat': lat, 'ID_trip': 7})]
            case_kinds = {'empty_emission'}
    if len(trips) > 1 and rng.random() < 0.05:
        # the points of the first trip split around the second one
        first, second = trips[0], trips[1]
        half = len(first) // 2
        trips[:2] = [first.iloc[:half], second, first.iloc[half:]]
        case_kinds.add('non_contiguous')
    gps = pd.concat(trips, ignore_index=True)
    return Case(G_mm, trans, candidates, id2edges, edges2id, gps, alpha, radius, case_kinds)


def _random_csr(rng, n_rows, n_cols, values, dtype, density):
    indptr, indices, data = [0], [], []
    for _ in range(n_rows):
        cols = list(np.flatnonzero(rng.random(n_cols) < density))
        if cols and rng.random() < 0.1:
            cols.append(cols[rng.integers(len(cols))])      # a duplicate entry, summed by scipy
        indices += cols
        data += list(rng.choice(values, len(cols)))
        indptr.append(len(indices))
    return csr_matrix((np.array(data, dtype=dtype), np.array(indices, dtype=np.int32),
                       np.array(indptr, dtype=np.int32)), shape=(n_rows, n_cols))


def make_kernel_case(seed: int) -> KernelCase:
    rng = np.random.default_rng(seed)
    n_edges = int(rng.integers(1, 41))
    n_rows = int(rng.integers(1, 60))
    emission = _random_csr(rng, n_rows, n_edges, [0, 1, 2, 3, 100, 398], np.int16, rng.uniform(0.05, 0.6))
    trans = _random_csr(rng, n_edges, n_edges, [0, 1, 50, 100], np.int8, rng.uniform(0.05, 0.9))
    n_states = int(rng.integers(1, min(n_edges, 30) + 1))
    sub_edges = list(set(int(e) for e in rng.choice(n_edges, n_states, replace=False)))
    start = int(rng.integers(n_rows))
    n_obs = int(rng.integers(1, n_rows - start + 1))
    return KernelCase(emission, trans, sub_edges, start, n_obs)


def geohashes(lon, lat):
    return pygeohash_fast.encode_many(np.asarray(lon), np.asarray(lat), 8)
