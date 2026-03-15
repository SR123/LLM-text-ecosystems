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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_theorem2_exact_soft_policy_followup as theorem2_soft  # noqa: E402

from drift_selection.checkpoints import atomic_save_json, atomic_save_text  # noqa: E402
from drift_selection.ngram_regeneration_lab import (  # noqa: E402
    LatentGrammarConfig,
    UrtextConfig,
    build_empirical_model,
    build_reference_rgram_utility,
    build_urtext,
)
from drift_selection.utils import ensure_dir, timestamp  # noqa: E402


VERSION = "V0_01"
OUTPUT_ROOT = GH_ROOT / "data" / "outputs" / "theorem3_cross_entropy_inheritance"
APPENDIX_FIGURE_PDF = GH_ROOT / "appendix" / "figures" / "generated" / "figure_appx_theorem3_cross_entropy_inheritance.pdf"
APPENDIX_FIGURE_PNG = GH_ROOT / "appendix" / "figures" / "generated" / "figure_appx_theorem3_cross_entropy_inheritance.png"
APPENDIX_TABLE = GH_ROOT / "appendix" / "tables" / "table_appx_theorem3_cross_entropy_results.tex"

torch.set_num_threads(1)


@dataclass(frozen=True)
class EnvironmentConfig:
    vocab_size: int = 10
    urtext_length: int = 5000
    max_order: int = 3
    top_k: int = 500
    beta: float = 3.0
    base_seed: int = 12345
    urtext_mode: str = "synthetic_iid"


@dataclass(frozen=True)
class NeuralConfig:
    embedding_dim: int = 32
    hidden_dim: int = 128
    learning_rate: float = 5e-3
    weight_decay: float = 1e-4

    def epochs_for_sample_size(self, sample_size: int) -> int:
        if sample_size <= 1000:
            return 140
        if sample_size <= 5000:
            return 180
        if sample_size <= 20000:
            return 220
        return 250

    def batch_size_for_sample_size(self, sample_size: int) -> int:
        if sample_size <= 2000:
            return min(256, sample_size)
        if sample_size <= 10000:
            return 512
        return 1024


@dataclass(frozen=True)
class ExperimentConfig:
    version: str = VERSION
    sample_sizes: tuple[int, ...] = (500, 1000, 2000, 5000, 10000, 20000, 50000)
    sample_seeds: tuple[int, ...] = (11, 17, 29)
    count_smoothing: float = 1e-3


@dataclass
class EnvironmentBundle:
    config: EnvironmentConfig
    contexts: list[tuple[int, ...]]
    context_tensor: torch.Tensor
    stationary: torch.Tensor
    q_tensor: torch.Tensor
    p_tensor: torch.Tensor
    representative_index: int
    representative_context: tuple[int, ...]
    utility_count: int
    base_kl_bits: float
    base_top1_match_rate: float
    manifest: dict[str, Any]
    state_to_index: dict[tuple[int, ...], int]
    cumulative_stationary: list[float]
    cumulative_q: list[list[tuple[int, float]]]


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


def kl_bits(q: dict[int, float], p: dict[int, float]) -> float:
    total = 0.0
    for token, q_prob in q.items():
        if q_prob <= 0.0:
            continue
        p_prob = float(p.get(token, 0.0))
        if p_prob <= 0.0:
            return float("inf")
        total += float(q_prob) * math.log2(float(q_prob) / p_prob)
    return float(total)


def top1_match_rate(q_tensor: torch.Tensor, pred_tensor: torch.Tensor, weights: torch.Tensor) -> float:
    q_top = torch.argmax(q_tensor, dim=1)
    p_top = torch.argmax(pred_tensor, dim=1)
    return float((weights * (q_top == p_top).double()).sum().item())


def exact_weighted_kl(q_tensor: torch.Tensor, pred_tensor: torch.Tensor, weights: torch.Tensor) -> float:
    eps = 1e-12
    safe_q = q_tensor.clamp_min(eps)
    safe_p = pred_tensor.clamp_min(eps)
    kl = safe_q * (torch.log2(safe_q) - torch.log2(safe_p))
    return float((weights[:, None] * kl).sum().item())


def cumulative_distribution(probs: list[float]) -> list[float]:
    total = 0.0
    cumulative: list[float] = []
    for prob in probs:
        total += float(prob)
        cumulative.append(total)
    if cumulative:
        cumulative[-1] = 1.0
    return cumulative


def sample_from_cumulative(cumulative: list[float], rng: random.Random) -> int:
    draw = rng.random()
    idx = 0
    while idx < len(cumulative) and cumulative[idx] < draw:
        idx += 1
    return min(idx, len(cumulative) - 1)


def power_stationary(q_tensor: torch.Tensor, contexts: list[tuple[int, ...]], state_to_index: dict[tuple[int, ...], int]) -> torch.Tensor:
    transitions = torch.zeros((len(contexts), len(contexts)), dtype=torch.float64)
    for index, context in enumerate(contexts):
        for token_index, prob in enumerate(q_tensor[index].tolist(), start=1):
            if prob <= 0.0:
                continue
            next_state = (context[1], int(token_index))
            transitions[index, state_to_index[next_state]] += float(prob)
    stationary = torch.full((len(contexts),), 1.0 / len(contexts), dtype=torch.float64)
    for _ in range(5000):
        updated = stationary @ transitions
        if torch.max(torch.abs(updated - stationary)).item() < 1e-13:
            stationary = updated
            break
        stationary = updated
    stationary = stationary / stationary.sum()
    return stationary


def build_environment_bundle(config: EnvironmentConfig) -> EnvironmentBundle:
    urtext_config = UrtextConfig(mode=config.urtext_mode, vocab_size=config.vocab_size)
    latent_config = LatentGrammarConfig(
        max_order=config.max_order,
        order_keep_probs=None,
        exact_support=False,
        support_cache_limit=8192,
    )
    bundle = build_urtext(urtext_config, latent_config, config.urtext_length, seed=config.base_seed)
    urtext_tokens = [int(token) for token in bundle.token_ids[: config.urtext_length]]
    utility = build_reference_rgram_utility(
        urtext_tokens,
        span=5,
        min_count=1,
        max_patterns=config.top_k,
        label=f"urtext_top{config.top_k}",
        unseen_category="undesirable",
    )
    base_model = build_empirical_model(urtext_tokens, config.max_order)
    planner = theorem2_soft.SoftProbTiltPlanner(model=base_model, utility=utility, beta=config.beta)

    contexts = [
        (first, second)
        for first in range(1, config.vocab_size + 1)
        for second in range(1, config.vocab_size + 1)
    ]
    state_to_index = {context: index for index, context in enumerate(contexts)}
    p_tensor = torch.zeros((len(contexts), config.vocab_size), dtype=torch.float64)
    q_tensor = torch.zeros((len(contexts), config.vocab_size), dtype=torch.float64)
    q_dicts: list[dict[int, float]] = []
    p_dicts: list[dict[int, float]] = []

    for index, context in enumerate(contexts):
        p_dist, _ = base_model.distribution(context)
        q_dist = planner.first_token_summary(context).first_token_distribution
        p_dicts.append({int(token): float(prob) for token, prob in p_dist.items()})
        q_dicts.append({int(token): float(prob) for token, prob in q_dist.items()})
        for token, prob in p_dist.items():
            p_tensor[index, int(token) - 1] = float(prob)
        for token, prob in q_dist.items():
            q_tensor[index, int(token) - 1] = float(prob)

    stationary = power_stationary(q_tensor, contexts, state_to_index)

    base_kl = 0.0
    base_top1 = 0.0
    rep_score = -1.0
    rep_index = 0
    for index, (q_dist, p_dist) in enumerate(zip(q_dicts, p_dicts)):
        ctx_kl = kl_bits(q_dist, p_dist)
        weight = float(stationary[index])
        base_kl += weight * ctx_kl
        q_top = min(token for token, prob in q_dist.items() if prob == max(q_dist.values()))
        p_top = min(token for token, prob in p_dist.items() if prob == max(p_dist.values()))
        base_top1 += weight * (1.0 if q_top == p_top else 0.0)
        score = weight * ctx_kl
        if score > rep_score:
            rep_score = score
            rep_index = index

    manifest = {
        "version": VERSION,
        "created_at": timestamp(),
        "environment_config": config.__dict__,
        "notes": [
            "Environment q is an exact depth-5 soft lookahead first-token law built from a synthetic trigram base model p.",
            "Desirable continuations are the top-K distinct 5-grams from the urtext U0.",
            "Training data for theorem 3 consists of i.i.d. (context, next-token) pairs with contexts sampled from the stationary context distribution of q and targets drawn from q(.|c).",
            "The count model uses a tiny symmetric pseudo-count solely to avoid infinite finite-sample KL from zero-count cells.",
        ],
    }
    context_tensor = torch.tensor([[context[0] - 1, context[1] - 1] for context in contexts], dtype=torch.long)
    cumulative_q: list[list[tuple[int, float]]] = []
    for index in range(len(contexts)):
        probs = q_tensor[index].tolist()
        cumulative = cumulative_distribution(probs)
        cumulative_q.append([(token_index, cutoff) for token_index, cutoff in enumerate(cumulative)])
    return EnvironmentBundle(
        config=config,
        contexts=contexts,
        context_tensor=context_tensor,
        stationary=stationary,
        q_tensor=q_tensor,
        p_tensor=p_tensor,
        representative_index=rep_index,
        representative_context=contexts[rep_index],
        utility_count=len(utility.desirable_set),
        base_kl_bits=float(base_kl),
        base_top1_match_rate=float(base_top1),
        manifest=manifest,
        state_to_index=state_to_index,
        cumulative_stationary=cumulative_distribution(stationary.tolist()),
        cumulative_q=cumulative_q,
    )


def sample_iid_pairs(environment: EnvironmentBundle, sample_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    state_ids: list[int] = []
    targets: list[int] = []
    for _ in range(sample_size):
        state_index = sample_from_cumulative(environment.cumulative_stationary, rng)
        state_ids.append(state_index)
        token_cutoffs = environment.cumulative_q[state_index]
        draw = rng.random()
        chosen = 0
        while chosen < len(token_cutoffs) and token_cutoffs[chosen][1] < draw:
            chosen += 1
        targets.append(min(chosen, len(token_cutoffs) - 1))
    state_tensor = torch.tensor(state_ids, dtype=torch.long)
    x_tensor = environment.context_tensor[state_tensor]
    y_tensor = torch.tensor(targets, dtype=torch.long)
    return state_tensor, x_tensor, y_tensor


def fit_count_model(
    environment: EnvironmentBundle,
    state_ids: torch.Tensor,
    targets: torch.Tensor,
    smoothing: float,
) -> torch.Tensor:
    counts = torch.full(
        (len(environment.contexts), environment.config.vocab_size),
        float(smoothing),
        dtype=torch.float64,
    )
    for state_index, target in zip(state_ids.tolist(), targets.tolist()):
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
    model = TinyNeuralNextTokenModel(
        vocab_size=environment.config.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
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


def mean_std_label(values: pd.Series) -> str:
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    return f"{mean:.3f} ± {std:.3f}"


def build_representative_distribution_rows(
    environment: EnvironmentBundle,
    count_predictions: list[torch.Tensor],
    neural_predictions: list[torch.Tensor],
) -> list[dict[str, Any]]:
    index = environment.representative_index
    mean_count = torch.stack(count_predictions, dim=0).mean(dim=0)[index]
    mean_neural = torch.stack(neural_predictions, dim=0).mean(dim=0)[index]
    q_row = environment.q_tensor[index]
    p_row = environment.p_tensor[index]
    rows: list[dict[str, Any]] = []
    for token in range(1, environment.config.vocab_size + 1):
        rows.append(
            {
                "context": " ".join(str(x) for x in environment.representative_context),
                "token": int(token),
                "base_p": float(p_row[token - 1]),
                "environment_q": float(q_row[token - 1]),
                "count_model_mean": float(mean_count[token - 1]),
                "neural_model_mean": float(mean_neural[token - 1]),
            }
        )
    return rows


def write_report(
    output_dir: Path,
    environment: EnvironmentBundle,
    aggregate_df: pd.DataFrame,
    rows_df: pd.DataFrame,
) -> None:
    final_rows = aggregate_df[aggregate_df["sample_size"] == aggregate_df["sample_size"].max()]
    lines = [
        f"# {VERSION} theorem-3 cross-entropy inheritance experiment",
        "",
        "## Evolved text environment",
        f"- Base model: synthetic trigram urtext (`V={environment.config.vocab_size}`, urtext length `{environment.config.urtext_length}`).",
        f"- Environment construction: exact depth-5 soft lookahead with `beta={environment.config.beta}`.",
        f"- Desirable set: top `{environment.config.top_k}` distinct 5-grams from the urtext.",
        f"- Exact weighted KL from base model to the evolved text environment: `{environment.base_kl_bits:.6f}` bits.",
        f"- Exact weighted top-1 match between base model and the evolved text environment: `{environment.base_top1_match_rate:.6f}`.",
        "",
        "## Training protocol",
        "- Training examples are i.i.d. `(context, next-token)` pairs.",
        "- Contexts are sampled from the stationary context distribution of the evolved text environment `q`.",
        "- Targets are sampled from the exact evolved-environment conditional `q(.|c)`.",
        "- Agents: a smoothed trigram count agent and a small neural next-token agent (token embeddings + one hidden layer, cross-entropy training).",
        "",
        "## Final 50k-pair results",
    ]
    for _, row in final_rows.iterrows():
        lines.append(
            f"- {row['learner_label']}: KL `{row['exact_kl_bits_mean']:.6f} ± {row['exact_kl_bits_std']:.6f}`, "
            f"top-1 `{row['top1_match_mean']:.6f} ± {row['top1_match_std']:.6f}`."
        )
    lines.extend([
        "",
        "## Files",
        f"- Raw per-seed results: `{output_dir / 'tables' / 'results_by_seed.csv'}`",
        f"- Aggregate summary: `{output_dir / 'tables' / 'results_aggregate.csv'}`",
        f"- Representative context distributions: `{output_dir / 'tables' / 'representative_context_distribution.csv'}`",
    ])
    atomic_save_text(output_dir / "report.md", "\n".join(lines))


def write_appendix_table(path: Path, environment: EnvironmentBundle, aggregate_df: pd.DataFrame) -> None:
    final_rows = aggregate_df[aggregate_df["sample_size"] == aggregate_df["sample_size"].max()].copy()
    count_row = final_rows[final_rows["learner"] == "count"].iloc[0]
    neural_row = final_rows[final_rows["learner"] == "neural"].iloc[0]
    table = rf"""\begin{{table}}[t]
\centering
\footnotesize
\caption{{Synthetic theorem~3 inheritance experiment: exact evolved-environment gap and downstream recovery.}}
\label{{tab:appx-theorem3-cross-entropy}}
\begin{{tabular}}{{lcc}}
\toprule
Model or environment & Exact KL to $q$ (bits) & Top-1 agreement with $q$ \\
\midrule
Base trigram kernel $p$ & {environment.base_kl_bits:.3f} & {environment.base_top1_match_rate:.3f} \\
Smoothed trigram agent (50,000 pairs) & {count_row['exact_kl_bits_mean']:.3f} $\pm$ {count_row['exact_kl_bits_std']:.3f} & {count_row['top1_match_mean']:.3f} $\pm$ {count_row['top1_match_std']:.3f} \\
Small neural next-token agent (50,000 pairs) & {neural_row['exact_kl_bits_mean']:.3f} $\pm$ {neural_row['exact_kl_bits_std']:.3f} & {neural_row['top1_match_mean']:.3f} $\pm$ {neural_row['top1_match_std']:.3f} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    atomic_save_text(path, table)


def plot_results(
    output_dir: Path,
    environment: EnvironmentBundle,
    aggregate_df: pd.DataFrame,
    representative_df: pd.DataFrame,
) -> tuple[Path, Path]:
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
    axes[0].axhline(environment.base_kl_bits, color="#555555", linestyle="--", linewidth=1.6, label="Base trigram kernel $p$")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Training pairs")
    axes[0].set_ylabel(r"Exact $\mathbb{E}_{c \sim \rho_q}\mathrm{KL}(q(\cdot\mid c)\,\|\,m(\cdot\mid c))$ (bits)")
    axes[0].set_title("Both agent classes approach the same evolved environment $q$")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    rep_context = representative_df["context"].iloc[0]
    representative_df = representative_df.copy()
    representative_df["environment_rank"] = representative_df["environment_q"].rank(method="first", ascending=False)
    plot_df = representative_df.sort_values("environment_rank").head(6)
    x = range(len(plot_df))
    width = 0.2
    axes[1].bar([i - 1.5 * width for i in x], plot_df["base_p"], width=width, label="Base $p$", color="#9e9e9e")
    axes[1].bar([i - 0.5 * width for i in x], plot_df["environment_q"], width=width, label="Environment $q$", color="#1565c0")
    axes[1].bar([i + 0.5 * width for i in x], plot_df["count_model_mean"], width=width, label="Trigram agent", color="#2e7d32")
    axes[1].bar([i + 1.5 * width for i in x], plot_df["neural_model_mean"], width=width, label="Neural agent", color="#c62828")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels([str(int(token)) for token in plot_df["token"]])
    axes[1].set_xlabel("Token")
    axes[1].set_ylabel("Probability")
    axes[1].set_title(f"Representative context c=({rep_context}) at 50k pairs")
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        f"{VERSION} | theorem-3 synthetic inheritance | V={environment.config.vocab_size}, "
        f"topK={environment.config.top_k}, beta={environment.config.beta} | {timestamp()}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)

    ensure_dir(output_dir / "figures")
    png_path = output_dir / "figures" / "theorem3_cross_entropy_inheritance.png"
    pdf_path = output_dir / "figures" / "theorem3_cross_entropy_inheritance.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> int:
    environment_config = EnvironmentConfig()
    neural_config = NeuralConfig()
    experiment_config = ExperimentConfig()

    run_name = (
        f"{VERSION.lower()}_environment_top{environment_config.top_k}_beta{str(environment_config.beta).replace('.', 'p')}_"
        f"{environment_config.urtext_mode}_v{environment_config.vocab_size}"
    )
    output_dir = OUTPUT_ROOT / run_name
    ensure_dir(output_dir)
    ensure_dir(output_dir / "tables")
    ensure_dir(output_dir / "figures")

    environment = build_environment_bundle(environment_config)
    manifest = {
        **environment.manifest,
        "experiment_config": experiment_config.__dict__,
        "neural_config": neural_config.__dict__,
        "run_name": run_name,
    }
    atomic_save_json(output_dir / "run_manifest.json", manifest)

    result_rows: list[dict[str, Any]] = []
    all_count_predictions: dict[int, list[torch.Tensor]] = {size: [] for size in experiment_config.sample_sizes}
    all_neural_predictions: dict[int, list[torch.Tensor]] = {size: [] for size in experiment_config.sample_sizes}

    progress = tqdm(
        total=len(experiment_config.sample_sizes) * len(experiment_config.sample_seeds),
        desc="theorem3_cross_entropy",
        unit="dataset",
    )
    for sample_size in experiment_config.sample_sizes:
        for sample_seed in experiment_config.sample_seeds:
            state_ids, x_tensor, y_tensor = sample_iid_pairs(environment, sample_size, sample_seed)

            count_predictions = fit_count_model(
                environment,
                state_ids,
                y_tensor,
                smoothing=experiment_config.count_smoothing,
            )
            count_kl = exact_weighted_kl(environment.q_tensor, count_predictions, environment.stationary)
            count_top1 = top1_match_rate(environment.q_tensor, count_predictions, environment.stationary)
            all_count_predictions[sample_size].append(count_predictions)
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

            neural_predictions, final_loss, epochs = fit_neural_model(
                environment,
                x_tensor,
                y_tensor,
                seed=1000 + int(sample_seed),
                config=neural_config,
            )
            neural_kl = exact_weighted_kl(environment.q_tensor, neural_predictions, environment.stationary)
            neural_top1 = top1_match_rate(environment.q_tensor, neural_predictions, environment.stationary)
            all_neural_predictions[sample_size].append(neural_predictions)
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

    representative_df = pd.DataFrame(
        build_representative_distribution_rows(
            environment,
            all_count_predictions[max(experiment_config.sample_sizes)],
            all_neural_predictions[max(experiment_config.sample_sizes)],
        )
    )

    results_df.to_csv(output_dir / "tables" / "results_by_seed.csv", index=False)
    aggregate_df.to_csv(output_dir / "tables" / "results_aggregate.csv", index=False)
    representative_df.to_csv(output_dir / "tables" / "representative_context_distribution.csv", index=False)

    environment_rows: list[dict[str, Any]] = []
    for index, context in enumerate(environment.contexts):
        row = {
            "context": " ".join(str(x) for x in context),
            "stationary_weight": float(environment.stationary[index]),
        }
        for token in range(1, environment.config.vocab_size + 1):
            row[f"p_token_{token}"] = float(environment.p_tensor[index, token - 1])
            row[f"q_token_{token}"] = float(environment.q_tensor[index, token - 1])
        environment_rows.append(row)
    pd.DataFrame(environment_rows).to_csv(output_dir / "tables" / "environment_kernel.csv", index=False)

    png_path, pdf_path = plot_results(output_dir, environment, aggregate_df, representative_df)
    ensure_dir(APPENDIX_FIGURE_PDF.parent)
    ensure_dir(APPENDIX_FIGURE_PNG.parent)
    shutil.copy2(pdf_path, APPENDIX_FIGURE_PDF)
    shutil.copy2(png_path, APPENDIX_FIGURE_PNG)
    write_appendix_table(APPENDIX_TABLE, environment, aggregate_df)
    write_report(output_dir, environment, aggregate_df, results_df)

    summary = {
        "run_name": run_name,
        "version": VERSION,
        "base_kl_bits": environment.base_kl_bits,
        "base_top1_match_rate": environment.base_top1_match_rate,
        "representative_context": " ".join(str(x) for x in environment.representative_context),
        "results_csv": str(output_dir / "tables" / "results_by_seed.csv"),
        "aggregate_csv": str(output_dir / "tables" / "results_aggregate.csv"),
        "figure_pdf": str(APPENDIX_FIGURE_PDF),
        "figure_png": str(APPENDIX_FIGURE_PNG),
        "appendix_table": str(APPENDIX_TABLE),
        "finished_at": timestamp(),
    }
    atomic_save_json(output_dir / "run_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
