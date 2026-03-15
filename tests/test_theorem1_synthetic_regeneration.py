from pathlib import Path
import random
import sys
import tempfile

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from drift_selection.theorem1_synthetic_regeneration import (
    RegenerationConfig,
    SyntheticGrammarConfig,
    next_generation_text,
    run_synthetic_regeneration,
)
from drift_selection.theorem2_selected_publication import SyntheticBackoffGrammar


def test_next_generation_text_preserves_requested_length():
    grammar = SyntheticBackoffGrammar(
        SyntheticGrammarConfig(vocab_size=20, keep_prob_2gram=0.4, keep_prob_3gram=0.15)
    )
    tokens, _ = grammar.generate(60)
    out = next_generation_text(
        tokens,
        RegenerationConfig(text_length=60, generations=3, alpha=0.25, restart_probability=0.0),
        rng=random.Random(5),
    )
    assert len(out) == 60


def test_run_synthetic_regeneration_writes_artifacts_and_resumes():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_root = Path(tmpdir)
        run = run_synthetic_regeneration(
            version="V0_01",
            run_name="regen_smoke",
            grammar_config=SyntheticGrammarConfig(vocab_size=30, keep_prob_2gram=0.35, keep_prob_3gram=0.12),
            regeneration_config=RegenerationConfig(text_length=80, generations=3, alpha=0.25, restart_probability=0.0),
            seed=11,
            output_root=output_root,
            resume=True,
            progress_bar=False,
        )
        assert len(run.metrics_rows) == 4
        assert run.paths.checkpoint_path.exists()
        assert run.paths.metrics_final_path.exists()
        assert run.paths.summary_path.exists()
        assert run.paths.sample_text_path.exists()
        assert len(run.final_tokens) == 80

        resumed = run_synthetic_regeneration(
            version="V0_01",
            run_name="regen_smoke",
            grammar_config=SyntheticGrammarConfig(vocab_size=30, keep_prob_2gram=0.35, keep_prob_3gram=0.12),
            regeneration_config=RegenerationConfig(text_length=80, generations=3, alpha=0.25, restart_probability=0.0),
            seed=11,
            output_root=output_root,
            resume=True,
            progress_bar=False,
        )
        assert resumed.manifest["status"] == "finished"
        assert len(resumed.metrics_rows) == len(run.metrics_rows)
        assert resumed.final_tokens == run.final_tokens
