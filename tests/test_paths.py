import networkx as nx
import pytest

from gpsmatcher.map_matching import _shortest_path


def test_cached_path_is_networkx_path():
    graph = nx.DiGraph([(1, 2), (2, 3), (1, 4), (4, 3)])
    paths = {}
    assert _shortest_path(graph, 1, 3, paths) == nx.shortest_path(graph, 1, 3)
    assert paths == {(1, 3): nx.shortest_path(graph, 1, 3)}
    assert _shortest_path(graph, 1, 3, paths) is paths[(1, 3)]


def test_no_cache_without_dict():
    graph = nx.DiGraph([(1, 2)])
    assert _shortest_path(graph, 1, 2, None) == [1, 2]


def test_missing_path_raises_every_time():
    graph = nx.DiGraph([(1, 2), (3, 4)])
    paths = {}
    for _ in range(2):
        with pytest.raises(nx.NetworkXNoPath):
            _shortest_path(graph, 1, 4, paths)
    assert paths == {}
