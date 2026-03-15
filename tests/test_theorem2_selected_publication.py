from pathlib import Path
import random
import sys
import tempfile

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from drift_selection.theorem2_selected_publication import (
    CandidateBatch,
    CountBackoffModel,
    ExperimentConfig,
    PatternLibrary,
    PublicationPolicy,
    SequenceRecord,
    SyntheticBackoffGrammar,
    SyntheticGrammarConfig,
    apply_publication_policy,
    backtracking_completion,
    run_publication_experiment,
    run_publication_comparison,
    sample_pattern_library,
)


def test_synthetic_grammar_generate_length_and_counts():
    grammar = SyntheticBackoffGrammar(
        SyntheticGrammarConfig(vocab_size=40, keep_prob_2gram=0.25, keep_prob_3gram=0.10)
    )
    tokens, order_counts = grammar.generate(length=25)
    assert len(tokens) == 25
    assert sum(order_counts) == 25
    assert order_counts[2] > 0


def test_apply_publication_policy_respects_category_probabilities():
    batch = CandidateBatch(
        round_index=1,
        records=(
            SequenceRecord(tokens=(1, 2, 3), order_counts=(0, 0, 3), desirable_hits=1, undesirable_hits=0, category="desirable"),
            SequenceRecord(tokens=(3, 2, 1), order_counts=(0, 0, 3), desirable_hits=0, undesirable_hits=1, category="undesirable"),
            SequenceRecord(tokens=(4, 5, 6), order_counts=(0, 0, 3), desirable_hits=0, undesirable_hits=0, category="neutral"),
        ),
    )
    policy = PublicationPolicy(
        name="selected",
        publish_neutral_prob=0.0,
        publish_desirable_prob=1.0,
        publish_undesirable_prob=0.0,
        publish_mixed_prob=0.0,
    )
    published_sequences, published_records = apply_publication_policy(
        batch,
        policy,
        random.Random(123),
    )
    assert published_sequences == [(1, 2, 3)]
    assert [record.category for record in published_records] == ["desirable"]


def test_backtracking_completion_finds_desirable_sequence_without_bad_pattern():
    patterns = PatternLibrary(
        span=3,
        desirable=((1, 2, 3),),
        undesirable=((1, 1, 1),),
    )
    model = CountBackoffModel(vocab_size=4, order=3)
    model.fit_sequences(
        [
            (1, 2, 3, 4),
            (2, 1, 2, 3),
            (1, 2, 3, 2),
        ]
    )
    sequence = backtracking_completion(
        model,
        patterns,
        total_length=4,
        prefix=(1, 2),
        branch_factor=3,
        max_expansions=50,
    )
    assert sequence is not None
    good_hits, bad_hits, _ = patterns.classify(sequence)
    assert good_hits > 0
    assert bad_hits == 0


def test_run_publication_comparison_returns_rows_for_each_policy_and_round():
    grammar = SyntheticBackoffGrammar(
        SyntheticGrammarConfig(vocab_size=30, keep_prob_2gram=0.30, keep_prob_3gram=0.12)
    )
    patterns = sample_pattern_library(
        grammar,
        span=4,
        desirable_count=1,
        undesirable_count=1,
        pool_sequences=150,
        sequence_length=16,
        seed=7,
    )
    config = ExperimentConfig(
        sequence_length=16,
        rounds=2,
        candidate_count=18,
        evaluation_count=10,
        search_trials=5,
        prefix_length=2,
        max_expansions=200,
    )
    result = run_publication_comparison(
        grammar,
        patterns,
        policies=(
            PublicationPolicy(name="neutral", publish_neutral_prob=0.4, publish_desirable_prob=0.4, publish_undesirable_prob=0.4, publish_mixed_prob=0.4),
            PublicationPolicy(name="selected", publish_neutral_prob=0.4, publish_desirable_prob=1.0, publish_undesirable_prob=0.0, publish_mixed_prob=0.0),
        ),
        config=config,
        seed=9,
    )
    rows = result.table_rows()
    assert len(rows) == 4
    assert set(result.trajectories) == {"neutral", "selected"}
    assert {row["round"] for row in rows} == {1, 2}


def test_run_publication_experiment_writes_artifacts_and_can_resume():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_root = Path(tmpdir)
        run = run_publication_experiment(
            version="V0_01",
            run_name="smoke_resume",
            grammar_config=SyntheticGrammarConfig(vocab_size=25, keep_prob_2gram=0.30, keep_prob_3gram=0.12),
            experiment_config=ExperimentConfig(
                sequence_length=14,
                rounds=2,
                candidate_count=12,
                evaluation_count=6,
                search_trials=4,
                prefix_length=2,
                max_expansions=120,
            ),
            policies=(
                PublicationPolicy(name="publish_all", publish_neutral_prob=1.0, publish_desirable_prob=1.0, publish_undesirable_prob=1.0, publish_mixed_prob=1.0),
                PublicationPolicy(name="selected", publish_neutral_prob=1.0, publish_desirable_prob=1.0, publish_undesirable_prob=0.0, publish_mixed_prob=0.0),
            ),
            output_root=output_root,
            pattern_sampler_kwargs={
                "span": 4,
                "desirable_count": 1,
                "undesirable_count": 1,
                "pool_sequences": 120,
                "sequence_length": 14,
            },
            seed=21,
            resume=True,
            progress_bar=False,
        )
        assert run.paths.checkpoint_path.exists()
        assert run.paths.metrics_final_path.exists()
        assert run.paths.summary_path.exists()
        assert run.paths.samples_path.exists()

        resumed = run_publication_experiment(
            version="V0_01",
            run_name="smoke_resume",
            grammar_config=SyntheticGrammarConfig(vocab_size=25, keep_prob_2gram=0.30, keep_prob_3gram=0.12),
            experiment_config=ExperimentConfig(
                sequence_length=14,
                rounds=2,
                candidate_count=12,
                evaluation_count=6,
                search_trials=4,
                prefix_length=2,
                max_expansions=120,
            ),
            policies=(
                PublicationPolicy(name="publish_all", publish_neutral_prob=1.0, publish_desirable_prob=1.0, publish_undesirable_prob=1.0, publish_mixed_prob=1.0),
                PublicationPolicy(name="selected", publish_neutral_prob=1.0, publish_desirable_prob=1.0, publish_undesirable_prob=0.0, publish_mixed_prob=0.0),
            ),
            output_root=output_root,
            seed=21,
            resume=True,
            progress_bar=False,
        )
        assert resumed.manifest["status"] == "finished"
        assert len(resumed.table_rows) == len(run.table_rows)
