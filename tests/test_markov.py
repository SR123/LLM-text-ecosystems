from pathlib import Path
import sys
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from drift_selection.markov import survival_probabilities, doob_h_transform


def test_survival_shape_and_bounds():
    P = np.array([[0.7, 0.3], [0.1, 0.9]], dtype=float)
    h = survival_probabilities(P, [True, True], horizon=5)
    assert h.shape == (6, 2)
    assert (h >= 0).all()
    assert (h <= 1.0 + 1e-9).all()


def test_doob_rows_sum_to_one():
    P = np.array([[0.7, 0.3], [0.2, 0.8]], dtype=float)
    h = np.array([0.5, 0.7])
    T = doob_h_transform(P, h, [True, True])
    assert np.allclose(T.sum(axis=1), np.array([1.0, 1.0]))
