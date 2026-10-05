import tomllib
from pathlib import Path

import numba
import pytest

import gpsmatcher.viterbi as viterbi


def test_shapely_2_is_declared():
    """emission.py calls shapely 2's vectorized functions (get_x, get_y, bounds, distance)."""
    project = tomllib.loads((Path(__file__).parents[1] / 'pyproject.toml').read_text())['project']
    assert any(dep.replace(' ', '') == 'shapely>=2' for dep in project['dependencies'])


def test_jit_without_a_writable_cache(monkeypatch):
    """Where numba finds no cache directory it can write, the kernel is compiled without the cache."""
    njit = numba.njit

    def no_locator(*args, **kwargs):
        if kwargs.get('cache'):
            raise RuntimeError("cannot cache function 'f': no locator available for file 'f.py'")
        return njit(*args, **kwargs)

    monkeypatch.setattr(numba, 'njit', no_locator)

    def double(x):
        return 2 * x

    assert viterbi._jit(double)(21) == 42
