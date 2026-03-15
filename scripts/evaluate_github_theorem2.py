#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
GH_ROOT = SCRIPT_DIR.parent
SRC_ROOT = GH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drift_selection.database import export_registry_csv, register_run
from drift_selection.github_theorem2 import ensure_github_dirs, load_yaml_config, read_csv, resolve_path, save_yaml_config, write_csv
from drift_selection.utils import timestamp


def _selected_test_ce(rows: list[dict[str, Any]], learner: str, model_name: str) -> float:
    vals = [float(r.get("cross_entropy", "nan")) for r in rows if r.get("learner_class") == learner and r.get("model_name") == model_name and r.get("eval_env") == "selected_test"]
    if not vals:
        return float("inf")
    return float(sum(vals) / len(vals))


def _load_json_or_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    from drift_selection.github_theorem2 import load_yaml_config

    return load_yaml_config(path)


def _fmt(v: float) -> str:
    if v != v:
        return "nan"
    if v == float("inf"):
        return "inf"
    return f"{v:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate and evaluate GitHub theorem-2 analogue results")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="GitHub/configs/github_theorem2_v1.yaml")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    cfg = load_yaml_config(resolve_path(root, args.config))
    dirs = ensure_github_dirs(root, cfg)

    output_root = dirs["output_root"]
    artifacts = cfg.get("artifacts", {})

    observational_csv = output_root / artifacts.get("observational_summary_csv", "github_observational_summary.csv")
    ngram_metrics = output_root / "metrics" / "ngram_metrics.csv"
    transformer_metrics = output_root / "metrics" / "transformer_metrics.csv"

    obs_rows = read_csv(observational_csv)
    ngram_rows = read_csv(ngram_metrics) if ngram_metrics.exists() else []
    transformer_rows = read_csv(transformer_metrics) if transformer_metrics.exists() else []

    all_rows = ngram_rows + transformer_rows
    if not all_rows:
        raise FileNotFoundError("No learner metrics found. Run n-gram and transformer training scripts first.")

    theorem2_csv = output_root / artifacts.get("theorem2_metrics_csv", "github_theorem2_metrics.csv")
    fieldnames: list[str] = []
    for row in all_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    write_csv(theorem2_csv, all_rows, fieldnames=fieldnames)

    ngram_candidate_ce = _selected_test_ce(all_rows, learner="learner_ngram", model_name="model_candidate")
    ngram_selected_ce = _selected_test_ce(all_rows, learner="learner_ngram", model_name="model_selected")
    tr_candidate_ce = _selected_test_ce(all_rows, learner="learner_transformer", model_name="model_candidate")
    tr_selected_ce = _selected_test_ce(all_rows, learner="learner_transformer", model_name="model_selected")

    # Observational difference proxy
    cand_obs = next((r for r in obs_rows if r.get("environment") == "candidate_env"), {})
    sel_obs = next((r for r in obs_rows if r.get("environment") == "selected_env"), {})
    entropy_gap = float(sel_obs.get("token_entropy", 0.0)) - float(cand_obs.get("token_entropy", 0.0))
    distinct2_gap = float(sel_obs.get("distinct_2", 0.0)) - float(cand_obs.get("distinct_2", 0.0))
    rep4_gap = float(sel_obs.get("repetition_4", 0.0)) - float(cand_obs.get("repetition_4", 0.0))

    corpora_manifest = _load_json_or_yaml(output_root / "manifests" / "corpora_build_manifest.yaml")
    split_manifest = _load_json_or_yaml(output_root / "manifests" / artifacts.get("split_manifest_yaml", "github_split_manifest.yaml"))
    ngram_manifest = _load_json_or_yaml(output_root / "manifests" / "ngram_training_manifest.yaml")
    transformer_manifest = _load_json_or_yaml(output_root / "manifests" / "transformer_training_manifest.json")

    stop_go = cfg.get("stop_go", {})
    min_jsd = float(stop_go.get("min_selected_candidate_jsd", 0.01))
    jsd = float(corpora_manifest.get("js_divergence_candidate_selected", 0.0) or 0.0)

    pass_observational = jsd >= min_jsd
    pass_ngram = ngram_selected_ce < ngram_candidate_ce
    pass_transformer = tr_selected_ce < tr_candidate_ce
    overall_pass = pass_observational and pass_ngram and pass_transformer

    success_rows = [
        {
            "check": "observational_difference_jsd",
            "value": jsd,
            "threshold": min_jsd,
            "pass": pass_observational,
        },
        {
            "check": "ngram_selected_better_on_selected_test",
            "value": ngram_candidate_ce - ngram_selected_ce,
            "threshold": 0.0,
            "pass": pass_ngram,
        },
        {
            "check": "transformer_selected_better_on_selected_test",
            "value": tr_candidate_ce - tr_selected_ce,
            "threshold": 0.0,
            "pass": pass_transformer,
        },
        {
            "check": "overall_theorem2_analogue",
            "value": 1.0 if overall_pass else 0.0,
            "threshold": 1.0,
            "pass": overall_pass,
        },
    ]
    write_csv(output_root / "metrics" / "github_success_checks.csv", success_rows)

    run_manifest = {
        "run_family": cfg.get("run_family", "github_theorem2_v1"),
        "timestamp": timestamp(),
        "labels": {
            "candidate_env": "candidate_env",
            "merged_env": "merged_env",
            "selected_env": "selected_env",
            "selected_test": "selected_test",
            "model_candidate": "model_candidate",
            "model_selected": "model_selected",
            "learner_ngram": "learner_ngram",
            "learner_transformer": "learner_transformer",
        },
        "budget_matching": {
            "tokenizer_shared": bool(transformer_manifest.get("matched_budget", {}).get("same_tokenizer", False)),
            "architecture_shared": bool(transformer_manifest.get("matched_budget", {}).get("same_architecture", False)),
            "steps_shared": bool(transformer_manifest.get("matched_budget", {}).get("same_steps", False)),
            "ngram_train_token_budget": ngram_manifest.get("train_token_budget"),
            "transformer_train_token_budget": transformer_manifest.get("matched_budget", {}).get("train_token_budget"),
        },
        "corpora": {
            "label_coverage": corpora_manifest.get("coverage", {}),
            "js_divergence_candidate_selected": jsd,
            "split_counts": split_manifest.get("counts", {}),
        },
        "primary_metrics": {
            "ngram": {
                "model_candidate_selected_test_ce": ngram_candidate_ce,
                "model_selected_selected_test_ce": ngram_selected_ce,
            },
            "transformer": {
                "model_candidate_selected_test_ce": tr_candidate_ce,
                "model_selected_selected_test_ce": tr_selected_ce,
            },
        },
        "success_checks": success_rows,
    }
    run_manifest_path = output_root / "manifests" / "run_manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")

    # repo/split manifest outputs required by user brief
    repo_manifest_yaml = output_root / "manifests" / artifacts.get("repo_manifest_yaml", "github_repo_manifest.yaml")
    if not repo_manifest_yaml.exists():
        fallback_repo_manifest = {
            "run_family": cfg.get("run_family", "github_theorem2_v1"),
            "created_at": timestamp(),
            "source": "config",
            "repos": cfg.get("collection", {}).get("repos", []),
            "num_repos": len(cfg.get("collection", {}).get("repos", [])),
        }
        save_yaml_config(repo_manifest_yaml, fallback_repo_manifest)

    split_manifest_yaml = output_root / "manifests" / artifacts.get("split_manifest_yaml", "github_split_manifest.yaml")
    if not split_manifest_yaml.exists() and split_manifest:
        save_yaml_config(split_manifest_yaml, split_manifest)

    report_path = output_root / artifacts.get("diagnostic_report_md", "diagnostic_github_report.md")
    report_lines = [
        "# GitHub Theorem-2 Diagnostic Report",
        "",
        "## Data collected",
        f"- Repo count (configured): {len(cfg.get('collection', {}).get('repos', []))}",
        f"- Candidate docs: {corpora_manifest.get('coverage', {}).get('candidate_env_docs', 'n/a')}",
        f"- Selected docs: {corpora_manifest.get('coverage', {}).get('selected_env_docs', 'n/a')}",
        "",
        "## Label coverage",
        f"- Coverage summary: `{json.dumps(corpora_manifest.get('coverage', {}), sort_keys=True)}`",
        "",
        "## Corpus sizes",
        f"- Split counts: `{json.dumps(split_manifest.get('counts', {}), sort_keys=True)}`",
        f"- JSD(candidate, selected): {jsd:.6f} (threshold {min_jsd:.6f})",
        "",
        "## Observational differences",
        f"- Entropy(selected - candidate): {entropy_gap:+.6f}",
        f"- Distinct-2(selected - candidate): {distinct2_gap:+.6f}",
        f"- Repetition-4(selected - candidate): {rep4_gap:+.6f}",
        "",
        "## Learner results",
        f"- N-gram selected_test CE: model_candidate={_fmt(ngram_candidate_ce)}, model_selected={_fmt(ngram_selected_ce)}",
        f"- Transformer selected_test CE: model_candidate={_fmt(tr_candidate_ce)}, model_selected={_fmt(tr_selected_ce)}",
        "",
        "## Does this support the theorem-2 analogue?",
        f"- Observational premise passed: {pass_observational}",
        f"- N-gram direction passed: {pass_ngram}",
        f"- Transformer direction passed: {pass_transformer}",
        f"- Overall: {overall_pass}",
        "",
        "## What to do next",
        "- If overall is false: inspect text samples and label rules, then tighten selected tier quality before retraining.",
        "- If n-gram passes but transformer fails: increase transformer budget modestly and verify tokenizer coverage.",
        "- If both pass: scale repos/time window and add confidence intervals by bootstrap over repositories.",
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    # register evaluation run
    run_id = register_run(
        db_path=dirs["database"],
        run_name="github_theorem2_evaluation",
        script_or_notebook="scripts/evaluate_github_theorem2.py",
        config_path=str(resolve_path(root, args.config)),
        metadata={
            "timestamp_start": timestamp(),
            "timestamp_end": timestamp(),
            "git_version": "unknown",
            "corpus": "github",
            "model_family": "aggregate",
            "selection_rule": "candidate_vs_selected",
            "seed": 0,
            "output_dir": str(output_root),
            "status": "completed",
            "metrics_summary": f"overall_pass={overall_pass}",
        },
    )
    _ = run_id

    registry_csv = output_root / "tables" / "experiment_registry.csv"
    export_registry_csv(dirs["database"], registry_csv)

    print(f"Wrote final metrics: {theorem2_csv}")
    print(f"Wrote run manifest: {run_manifest_path}")
    print(f"Wrote diagnostic report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
