from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from drift_selection.ngram import NgramModel, support_dropout_rate


def test_ngram_generate_length():
    m = NgramModel(order=3)
    tokens = "a b c a b d a b e".split()
    m.fit(tokens)
    out = m.generate(tokens[:2], length=20)
    assert len(out) == 20


def test_support_dropout_nonnegative():
    a = NgramModel(order=2)
    b = NgramModel(order=2)
    a.fit("a b a b a b".split())
    b.fit("a b".split())
    rate = support_dropout_rate(a, b)
    assert 0.0 <= rate <= 1.0
