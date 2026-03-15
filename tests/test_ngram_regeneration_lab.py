from pathlib import Path
import sys
import tempfile

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from drift_selection.ngram_regeneration_lab import (
    AgentConfig,
    LatentGrammarConfig,
    RegenerationLabConfig,
    SelectionConfig,
    UrtextConfig,
    build_frequency_rgram_utility,
    build_reference_rgram_utility,
    run_ngram_regeneration_lab,
)


def test_build_reference_rgram_utility_marks_seen_windows_as_desirable():
    utility = build_reference_rgram_utility(
        [1, 2, 3, 4, 1, 2, 3, 5],
        span=3,
        min_count=2,
    )
    assert utility.classify_window((1, 2, 3)) == "desirable"
    assert utility.classify_window((2, 3, 4)) == "neutral"


def test_build_frequency_rgram_utility_scores_repeated_windows_higher():
    utility = build_frequency_rgram_utility(
        [1, 2, 3, 1, 2, 3, 1, 2, 4],
        span=3,
        min_count=1,
        unseen_score=-0.25,
    )
    assert utility.score_window((1, 2, 3)) > utility.score_window((1, 2, 4))
    assert utility.classify_window((9, 9, 9)) == "undesirable"


def test_run_ngram_regeneration_lab_writes_artifacts_and_resumes():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_root = Path(tmpdir)
        run = run_ngram_regeneration_lab(
            version="V0_01",
            run_name="ngram_lab_smoke",
            urtext_config=UrtextConfig(mode="synthetic_latent", vocab_size=24),
            latent_config=LatentGrammarConfig(
                max_order=4,
                order_keep_probs={2: 0.50, 3: 0.25, 4: 0.125},
            ),
            lab_config=RegenerationLabConfig(
                text_length=90,
                generations=3,
                alpha=0.25,
                max_order=4,
                restart_probability=0.0,
            ),
            selection_config=SelectionConfig(
                mode="reference_frequency_rgram",
                span=4,
                lookahead_samples=12,
                reference_min_count=1,
                reference_unseen_score=-0.25,
            ),
            agent_config=AgentConfig(
                mode="tree_search",
                publish_strategy="best_score",
                lookahead_depth=4,
                branch_factor=4,
                max_expansions=120,
                publish_horizon=2,
                rollback_penalty=0.10,
            ),
            seed=17,
            output_root=output_root,
            resume=True,
            progress_bar=False,
        )
        assert len(run.metrics_rows) == 4
        assert run.paths.checkpoint_path.exists()
        assert run.paths.metrics_final_path.exists()
        assert run.paths.summary_path.exists()
        assert run.paths.sample_text_path.exists()
        assert len(run.final_tokens) == 90
        assert "desirable_windows" in run.metrics_rows[-1]
        assert "agent_decisions" in run.metrics_rows[-1]

        resumed = run_ngram_regeneration_lab(
            version="V0_01",
            run_name="ngram_lab_smoke",
            urtext_config=UrtextConfig(mode="synthetic_latent", vocab_size=24),
            latent_config=LatentGrammarConfig(
                max_order=4,
                order_keep_probs={2: 0.50, 3: 0.25, 4: 0.125},
            ),
            lab_config=RegenerationLabConfig(
                text_length=90,
                generations=3,
                alpha=0.25,
                max_order=4,
                restart_probability=0.0,
            ),
            selection_config=SelectionConfig(
                mode="reference_frequency_rgram",
                span=4,
                lookahead_samples=12,
                reference_min_count=1,
                reference_unseen_score=-0.25,
            ),
            agent_config=AgentConfig(
                mode="tree_search",
                publish_strategy="best_score",
                lookahead_depth=4,
                branch_factor=4,
                max_expansions=120,
                publish_horizon=2,
                rollback_penalty=0.10,
            ),
            seed=17,
            output_root=output_root,
            resume=True,
            progress_bar=False,
        )
        assert resumed.manifest["status"] == "finished"
        assert len(resumed.metrics_rows) == len(run.metrics_rows)
        assert resumed.final_tokens == run.final_tokens
