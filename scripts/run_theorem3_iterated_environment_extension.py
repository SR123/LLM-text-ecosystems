#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
GH_ROOT = SCRIPT_DIR.parent
SRC_ROOT = GH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drift_selection.checkpoints import atomic_save_json, atomic_save_text, load_pickle_checkpoint  # noqa: E402
from drift_selection.ngram_regeneration_lab import (  # noqa: E402
    LatentGrammarConfig,
    UrtextConfig,
    build_empirical_model,
    build_urtext,
)
from drift_selection.utils import ensure_dir, timestamp  # noqa: E402


VERSION = "V0_02"
OUTPUT_ROOT = GH_ROOT / "data" / "outputs" / "theorem3_iterated_environment_extension"
APPENDIX_FIGURE_PDF = GH_ROOT / "appendix" / "figures" / "generated" / "figure_appx_theorem3_iterated_environment_extension.pdf"
APPENDIX_FIGURE_PNG = GH_ROOT / "appendix" / "figures" / "generated" / "figure_appx_theorem3_iterated_environment_extension.png"
APPENDIX_TABLE = GH_ROOT / "appendix" / "tables" / "table_appx_theorem3_iterated_environment_extension.tex"

torch.set_num_threads(1)


@dataclass(frozen=True)
class UpstreamRunConfig:
    run_dir: str = "/Users/sorenriis/Documents/Collaboration/GitHub/data/outputs/theorem2_soft_policy_followup/runs/v0_01_theorem2_softfollow_soft_prob_beta_1_prevref_iid_uniform_v100_m1000_o3_g40_a0p1"


@dataclass(frozen=True)
class NeuralConfig:
    embedding_dim: int = 48
    hidden_dim: int = 192
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4

    def epochs_for_sample_size(self, sample_size: int) -> int:
        if sample_size <= 1000:
            return 100
        if sample_size <= 5000:
            return 120
        if sample_size <= 10000:
            return 140
        return 160

    def batch_size_for_sample_size(self, sample_size: int) -> int:
        if sample_size <= 2000:
            return min(256, sample_size)
        if sample_size <= 10000:
            return 512
        return 1024


@dataclass(frozen=True)
class ExperimentConfig:
    sample_sizes: tuple[int, ...] = (1000, 5000, 10000, 20000)
    sample_seeds: tuple[int, ...] = (11, 17, 29)
    count_smoothing: float = 1e-3


@dataclass
class EnvironmentBundle:
    vocab_size: int
    contexts: list[tuple[int, int]]
    context_tensor: torch.Tensor
    stationary: torch.Tensor
    q_tensor: torch.Tensor
    p0_tensor: torch.Tensor
    representative_context: tuple[int, int]
    representative_index: int
    baseline_kl_bits: float
    baseline_top1_match_rate: float
    cumulative_stationary: list[float]
    cumulative_q: list[list[float]]
    manifest: dict[str, Any]


class TinyNeuralNextTokenModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        return self.mlp(emb.reshape(x.shape[0], -1))


def cumulative_distribution(probs: list[float]) -> list[float]:
    total = 0.0
    out: list[float] = []
    for prob in probs:
        total += float(prob)
        out.append(total)
    if out:
        out[-1] = 1.0
    return out


def sample_from_cumulative(cumulative: list[float], rng: random.Random) -> int:
    draw = rng.random()
    index = 0
    while index < len(cumulative) and cumulative[index] < draw:
        index += 1
    return min(index, len(cumulative) - 1)


def exact_weighted_kl(q_tensor: torch.Tensor, pred_tensor: torch.Tensor, weights: torch.Tensor) -> float:
    eps = 1e-12
    safe_q = q_tensor.clamp_min(eps)
    safe_p = pred_tensor.clamp_min(eps)
    kl = safe_q * (torch.log2(safe_q) - torch.log2(safe_p))
    return float((weights[:, None] * kl).sum().item())


def top1_match_rate(q_tensor: torch.Tensor, pred_tensor: torch.Tensor, weights: torch.Tensor) -> float:
    q_top = torch.argmax(q_tensor, dim=1)
    p_top = torch.argmax(pred_tensor, dim=1)
    return float((weights * (q_top == p_top).double()).sum().item())


def power_stationary(q_tensor: torch.Tensor, contexts: list[tuple[int, int]]) -> torch.Tensor:
    vocab_size = int(q_tensor.shape[1])
    second_tokens = torch.tensor([context[1] - 1 for context in contexts], dtype=torch.long)
    token_indices = torch.arange(vocab_size, dtype=torch.long)
    target_indices = second_tokens[:, None] * vocab_size + token_indices[None, :]
    stationary = torch.full((len(contexts),), 1.0 / len(contexts), dtype=torch.float64)
    for _ in range(5000):
        updated = torch.zeros_like(stationary)
        updated.scatter_add_(
            0,
            target_indices.reshape(-1),
            (stationary[:, None] * q_tensor).reshape(-1),
        )
        if torch.max(torch.abs(updated - stationary)).item() < 1e-12:
            stationary = updated
            break
        stationary = updated
    return stationary / stationary.sum()


def reconstruct_initial_tokens(manifest: dict[str, Any]) -> list[int]:
    urtext_config = UrtextConfig(**manifest["urtext_config"])
    latent_config = LatentGrammarConfig(**manifest["latent_config"])
    text_length = int(manifest["lab_config"]["text_length"])
    seed = int(manifest["seed"])
    bundle = build_urtext(urtext_config, latent_config, text_length, seed=seed)
    return [int(token) for token in bundle.token_ids[:text_length]]


def build_environment_from_run(config: UpstreamRunConfig) -> EnvironmentBundle:
    run_dir = Path(config.run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    generations = int(manifest["lab_config"]["generations"])
    final_tokens = [int(token) for token in load_pickle_checkpoint(run_dir / "state" / f"tokens_gen_{generations:03d}.pkl")]
    initial_tokens = reconstruct_initial_tokens(manifest)

    final_model = build_empirical_model(final_tokens, int(manifest["lab_config"]["max_order"]))
    initial_model = build_empirical_model(initial_tokens, int(manifest["lab_config"]["max_order"]))
    vocab_size = int(manifest["urtext_config"]["vocab_size"])
    contexts = [(first, second) for first in range(1, vocab_size + 1) for second in range(1, vocab_size + 1)]
    q_tensor = torch.zeros((len(contexts), vocab_size), dtype=torch.float64)
    p0_tensor = torch.zeros((len(contexts), vocab_size), dtype=torch.float64)

    for index, context in enumerate(contexts):
        q_dist, _ = final_model.distribution(context)
        p0_dist, _ = initial_model.distribution(context)
        for token, prob in q_dist.items():
            q_tensor[index, int(token) - 1] = float(prob)
        for token, prob in p0_dist.items():
            p0_tensor[index, int(token) - 1] = float(prob)

    stationary = power_stationary(q_tensor, contexts)
    baseline_kl = exact_weighted_kl(q_tensor, p0_tensor, stationary)
    baseline_top1 = top1_match_rate(q_tensor, p0_tensor, stationary)

    rep_index = 0
    rep_score = -1.0
    eps = 1e-12
    for index in range(len(contexts)):
        q_row = q_tensor[index].clamp_min(eps)
        p_row = p0_tensor[index].clamp_min(eps)
        ctx_kl = float((q_row * (torch.log2(q_row) - torch.log2(p_row))).sum().item())
        score = float(stationary[index]) * ctx_kl
        if score > rep_score:
            rep_score = score
            rep_index = index

    manifest_out = {
        "version": VERSION,
        "created_at": timestamp(),
        "upstream_run_dir": str(run_dir),
        "upstream_policy_name": manifest.get("policy_name"),
        "upstream_policy_beta": manifest.get("policy_beta"),
        "upstream_lab_config": manifest.get("lab_config"),
        "upstream_urtext_config": manifest.get("urtext_config"),
        "notes": [
            "Environment q is the empirical trigram kernel of the final public corpus from a saved theorem-2 publication run.",
            "Baseline p0 is the empirical trigram kernel of the reconstructed initial urtext U0 for that same run.",
            "Downstream theorem-3 training data are i.i.d. (context, next-token) pairs sampled from the stationary context distribution and next-token law of the final public environment.",
        ],
    }
    return EnvironmentBundle(
        vocab_size=vocab_size,
        contexts=contexts,
        context_tensor=torch.tensor([[c[0] - 1, c[1] - 1] for c in contexts], dtype=torch.long),
        stationary=stationary,
        q_tensor=q_tensor,
        p0_tensor=p0_tensor,
        representative_context=contexts[rep_index],
        representative_index=rep_index,
        baseline_kl_bits=baseline_kl,
        baseline_top1_match_rate=baseline_top1,
        cumulative_stationary=cumulative_distribution(stationary.tolist()),
        cumulative_q=[cumulative_distribution(q_tensor[index].tolist()) for index in range(len(contexts))],
        manifest=manifest_out,
    )


def sample_iid_pairs(environment: EnvironmentBundle, sample_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    state_ids: list[int] = []
    targets: list[int] = []
    for _ in range(sample_size):
        state_index = sample_from_cumulative(environment.cumulative_stationary, rng)
        state_ids.append(state_index)
        targets.append(sample_from_cumulative(environment.cumulative_q[state_index], rng))
    state_tensor = torch.tensor(state_ids, dtype=torch.long)
    x_tensor = environment.context_tensor[state_tensor]
    y_tensor = torch.tensor(targets, dtype=torch.long)
    return state_tensor, x_tensor, y_tensor


def fit_count_model(environment: EnvironmentBundle, state_ids: torch.Tensor, y_tensor: torch.Tensor, smoothing: float) -> torch.Tensor:
    counts = torch.full((len(environment.contexts), environment.vocab_size), float(smoothing), dtype=torch.float64)
    for state_index, target in zip(state_ids.tolist(), y_tensor.tolist()):
        counts[int(state_index), int(target)] += 1.0
    return counts / counts.sum(dim=1, keepdim=True)


def fit_neural_model(
    environment: EnvironmentBundle,
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    seed: int,
    config: NeuralConfig,
) -> tuple[torch.Tensor, float, int]:
    torch.manual_seed(seed)
    model = TinyNeuralNextTokenModel(environment.vocab_size, config.embedding_dim, config.hidden_dim)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    epochs = config.epochs_for_sample_size(int(len(x_tensor)))
    batch_size = config.batch_size_for_sample_size(int(len(x_tensor)))
    final_loss = float("nan")
    for _ in range(epochs):
        permutation = torch.randperm(len(x_tensor))
        for start in range(0, len(x_tensor), batch_size):
            batch_index = permutation[start:start + batch_size]
            logits = model(x_tensor[batch_index])
            loss = nn.functional.cross_entropy(logits, y_tensor[batch_index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())
    with torch.no_grad():
        predictions = torch.softmax(model(environment.context_tensor), dim=-1).double()
    return predictions, final_loss, epochs


def build_representative_rows(
    environment: EnvironmentBundle,
    count_predictions: list[torch.Tensor],
    neural_predictions: list[torch.Tensor],
) -> pd.DataFrame:
    index = environment.representative_index
    count_mean = torch.stack(count_predictions, dim=0).mean(dim=0)[index]
    neural_mean = torch.stack(neural_predictions, dim=0).mean(dim=0)[index]
    q_row = environment.q_tensor[index]
    p0_row = environment.p0_tensor[index]
    rows: list[dict[str, Any]] = []
    for token in range(1, environment.vocab_size + 1):
        rows.append(
            {
                "context": " ".join(str(x) for x in environment.representative_context),
                "token": int(token),
                "initial_p0": float(p0_row[token - 1]),
                "final_q": float(q_row[token - 1]),
                "count_model_mean": float(count_mean[token - 1]),
                "neural_model_mean": float(neural_mean[token - 1]),
            }
        )
    return pd.DataFrame(rows)


def write_appendix_table(path: Path, environment: EnvironmentBundle, aggregate_df: pd.DataFrame) -> None:
    final_size = int(aggregate_df["sample_size"].max())
    final_rows = aggregate_df[aggregate_df["sample_size"] == final_size]
    count_row = final_rows[final_rows["learner"] == "count"].iloc[0]
    neural_row = final_rows[final_rows["learner"] == "neural"].iloc[0]
    table = rf"""\begin{{table}}[t]
\centering
\footnotesize
\caption{{Theorem~3 extension: downstream recovery from an actual iterated theorem~2 environment.}}
\label{{tab:appx-theorem3-iterated-extension}}
\begin{{tabular}}{{lcc}}
\toprule
Model or environment & Exact KL to final $q$ (bits) & Top-1 agreement with final $q$ \\
\midrule
Initial trigram kernel $p_0$ & {environment.baseline_kl_bits:.3f} & {environment.baseline_top1_match_rate:.3f} \\
Smoothed trigram agent ({final_size:,} pairs) & {count_row['exact_kl_bits_mean']:.3f} $\pm$ {count_row['exact_kl_bits_std']:.3f} & {count_row['top1_match_mean']:.3f} $\pm$ {count_row['top1_match_std']:.3f} \\
Small neural next-token agent ({final_size:,} pairs) & {neural_row['exact_kl_bits_mean']:.3f} $\pm$ {neural_row['exact_kl_bits_std']:.3f} & {neural_row['top1_match_mean']:.3f} $\pm$ {neural_row['top1_match_std']:.3f} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    atomic_save_text(path, table)


def plot_results(output_dir: Path, environment: EnvironmentBundle, aggregate_df: pd.DataFrame, representative_df: pd.DataFrame) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))

    for learner, color, label in [
        ("count", "#1b5e20", "Smoothed trigram agent"),
        ("neural", "#8e0000", "Small neural next-token agent"),
    ]:
        subset = aggregate_df[aggregate_df["learner"] == learner].sort_values("sample_size")
        x = subset["sample_size"].to_numpy()
        y = subset["exact_kl_bits_mean"].to_numpy()
        err = subset["exact_kl_bits_std"].to_numpy()
        axes[0].plot(x, y, marker="o", linewidth=2.0, color=color, label=label)
        axes[0].fill_between(x, y - err, y + err, alpha=0.15, color=color)
    axes[0].axhline(environment.baseline_kl_bits, color="#555555", linestyle="--", linewidth=1.6, label="Initial theorem-2 kernel $p_0$")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Training pairs from final theorem-2 environment")
    axes[0].set_ylabel(r"Exact $\mathbb{E}_{c \sim \rho_q}\mathrm{KL}(q(\cdot\mid c)\,\|\,m(\cdot\mid c))$ (bits)")
    axes[0].set_title("Downstream agents recover the final theorem-2 environment")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    rep_context = representative_df["context"].iloc[0]
    representative_df = representative_df.copy()
    representative_df["environment_rank"] = representative_df["final_q"].rank(method="first", ascending=False)
    plot_df = representative_df.sort_values("environment_rank").head(6)
    x = range(len(plot_df))
    width = 0.2
    axes[1].bar([i - 1.5 * width for i in x], plot_df["initial_p0"], width=width, label="Initial $p_0$", color="#9e9e9e")
    axes[1].bar([i - 0.5 * width for i in x], plot_df["final_q"], width=width, label="Final $q$", color="#1565c0")
    axes[1].bar([i + 0.5 * width for i in x], plot_df["count_model_mean"], width=width, label="Trigram agent", color="#2e7d32")
    axes[1].bar([i + 1.5 * width for i in x], plot_df["neural_model_mean"], width=width, label="Neural agent", color="#c62828")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels([str(int(token)) for token in plot_df["token"]])
    axes[1].set_xlabel("Token")
    axes[1].set_ylabel("Probability")
    axes[1].set_title(f"Representative context c=({rep_context}) at 20k pairs")
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"{VERSION} | theorem-3 iterated-environment extension | {timestamp()}", fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)

    ensure_dir(output_dir / "figures")
    png_path = output_dir / "figures" / "theorem3_iterated_environment_extension.png"
    pdf_path = output_dir / "figures" / "theorem3_iterated_environment_extension.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> int:
    upstream_config = UpstreamRunConfig()
    neural_config = NeuralConfig()
    experiment_config = ExperimentConfig()
    environment = build_environment_from_run(upstream_config)

    run_name = f"{VERSION.lower()}_iterated_env_from_{Path(upstream_config.run_dir).name}"
    output_dir = OUTPUT_ROOT / run_name
    ensure_dir(output_dir / "tables")
    ensure_dir(output_dir / "figures")

    atomic_save_json(
        output_dir / "run_manifest.json",
        {
            **environment.manifest,
            "version": VERSION,
            "run_name": run_name,
            "experiment_config": experiment_config.__dict__,
            "neural_config": neural_config.__dict__,
        },
    )

    result_rows: list[dict[str, Any]] = []
    all_count_predictions: dict[int, list[torch.Tensor]] = {size: [] for size in experiment_config.sample_sizes}
    all_neural_predictions: dict[int, list[torch.Tensor]] = {size: [] for size in experiment_config.sample_sizes}

    progress = tqdm(total=len(experiment_config.sample_sizes) * len(experiment_config.sample_seeds), desc="theorem3_iterated_ext", unit="dataset")
    for sample_size in experiment_config.sample_sizes:
        for sample_seed in experiment_config.sample_seeds:
            state_ids, x_tensor, y_tensor = sample_iid_pairs(environment, sample_size, sample_seed)

            count_pred = fit_count_model(environment, state_ids, y_tensor, smoothing=experiment_config.count_smoothing)
            count_kl = exact_weighted_kl(environment.q_tensor, count_pred, environment.stationary)
            count_top1 = top1_match_rate(environment.q_tensor, count_pred, environment.stationary)
            all_count_predictions[sample_size].append(count_pred)
            result_rows.append(
                {
                    "sample_size": int(sample_size),
                    "sample_seed": int(sample_seed),
                    "learner": "count",
                    "learner_label": "Smoothed trigram agent",
                    "exact_kl_bits": float(count_kl),
                    "top1_match_rate": float(count_top1),
                    "epochs": 0,
                    "final_train_loss": float("nan"),
                }
            )

            neural_pred, final_loss, epochs = fit_neural_model(
                environment,
                x_tensor,
                y_tensor,
                seed=1000 + int(sample_seed),
                config=neural_config,
            )
            neural_kl = exact_weighted_kl(environment.q_tensor, neural_pred, environment.stationary)
            neural_top1 = top1_match_rate(environment.q_tensor, neural_pred, environment.stationary)
            all_neural_predictions[sample_size].append(neural_pred)
            result_rows.append(
                {
                    "sample_size": int(sample_size),
                    "sample_seed": int(sample_seed),
                    "learner": "neural",
                    "learner_label": "Small neural next-token agent",
                    "exact_kl_bits": float(neural_kl),
                    "top1_match_rate": float(neural_top1),
                    "epochs": int(epochs),
                    "final_train_loss": float(final_loss),
                }
            )
            progress.update(1)
    progress.close()

    results_df = pd.DataFrame(result_rows).sort_values(["sample_size", "sample_seed", "learner"])
    aggregate_df = (
        results_df.groupby(["sample_size", "learner", "learner_label"], as_index=False)
        .agg(
            exact_kl_bits_mean=("exact_kl_bits", "mean"),
            exact_kl_bits_std=("exact_kl_bits", "std"),
            top1_match_mean=("top1_match_rate", "mean"),
            top1_match_std=("top1_match_rate", "std"),
        )
        .fillna(0.0)
    )
    representative_df = build_representative_rows(
        environment,
        all_count_predictions[max(experiment_config.sample_sizes)],
        all_neural_predictions[max(experiment_config.sample_sizes)],
    )

    results_df.to_csv(output_dir / "tables" / "results_by_seed.csv", index=False)
    aggregate_df.to_csv(output_dir / "tables" / "results_aggregate.csv", index=False)
    representative_df.to_csv(output_dir / "tables" / "representative_context_distribution.csv", index=False)

    png_path, pdf_path = plot_results(output_dir, environment, aggregate_df, representative_df)
    ensure_dir(APPENDIX_FIGURE_PDF.parent)
    ensure_dir(APPENDIX_FIGURE_PNG.parent)
    shutil.copy2(pdf_path, APPENDIX_FIGURE_PDF)
    shutil.copy2(png_path, APPENDIX_FIGURE_PNG)
    write_appendix_table(APPENDIX_TABLE, environment, aggregate_df)

    report_lines = [
        f"# {VERSION} theorem-3 iterated-environment extension",
        "",
        f"- Upstream theorem-2 run: `{upstream_config.run_dir}`",
        f"- Final-environment KL from initial kernel: `{environment.baseline_kl_bits:.6f}` bits.",
        f"- Final-environment top-1 match with initial kernel: `{environment.baseline_top1_match_rate:.6f}`.",
        "",
        "## Final 20k-pair results",
    ]
    final_rows = aggregate_df[aggregate_df["sample_size"] == aggregate_df["sample_size"].max()]
    for _, row in final_rows.iterrows():
        report_lines.append(
            f"- {row['learner_label']}: KL `{row['exact_kl_bits_mean']:.6f} ± {row['exact_kl_bits_std']:.6f}`, "
            f"top-1 `{row['top1_match_mean']:.6f} ± {row['top1_match_std']:.6f}`."
        )
    atomic_save_text(output_dir / "report.md", "\n".join(report_lines))

    atomic_save_json(
        output_dir / "run_summary.json",
        {
            "run_name": run_name,
            "version": VERSION,
            "baseline_kl_bits": environment.baseline_kl_bits,
            "baseline_top1_match_rate": environment.baseline_top1_match_rate,
            "results_csv": str(output_dir / "tables" / "results_by_seed.csv"),
            "aggregate_csv": str(output_dir / "tables" / "results_aggregate.csv"),
            "figure_pdf": str(APPENDIX_FIGURE_PDF),
            "figure_png": str(APPENDIX_FIGURE_PNG),
            "appendix_table": str(APPENDIX_TABLE),
            "finished_at": timestamp(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
