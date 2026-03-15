from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from drift_selection.selection import viability_selection, anti_viability_selection


def test_viability_selection_normalized():
    probs = {"A": 0.6, "B": 0.4}
    viability = {"A": 0.1, "B": 0.9}
    out = viability_selection(probs, viability)
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert out["B"] > out["A"]


def test_anti_viability_selection_normalized():
    probs = {"A": 0.6, "B": 0.4}
    viability = {"A": 0.9, "B": 0.1}
    out = anti_viability_selection(probs, viability)
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert out["B"] > out["A"]
