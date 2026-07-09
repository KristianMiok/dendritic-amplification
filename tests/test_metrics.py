"""Sanity tests for the standalone metric functions (these don't need data)."""
import numpy as np
from dendamp.degradation import schoener_d, warren_i, range_area_change


def test_identical_surfaces():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    assert abs(schoener_d(p, p) - 1.0) < 1e-9
    assert abs(warren_i(p, p) - 1.0) < 1e-9


def test_disjoint_surfaces():
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 0.0, 1.0])
    assert schoener_d(p, q) < 1e-9
    assert warren_i(p, q) < 1e-9


def test_range_inflation_sign():
    base = np.array([0.6, 0.6, 0.1, 0.1])          # 2 suitable
    infl = np.array([0.6, 0.6, 0.6, 0.1])          # 3 suitable
    assert range_area_change(base, infl, 0.5) == 0.5
