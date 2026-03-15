from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_figure_pdf_png(fig, out_path_base: Path) -> None:
    out_path_base = Path(out_path_base)
    _ensure_parent(out_path_base)
    fig.savefig(out_path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path_base.with_suffix(".png"), dpi=220, bbox_inches="tight")


def plot_teacher_student_bar_chart(
    metrics_df: pd.DataFrame,
    out_path_base: Path,
    title: str,
) -> None:
    cols = [
        c
        for c in [
            "mean_selected_probability",
            "top1_match_rate",
            "mean_branch_score",
            "support_rate_trigram",
        ]
        if c in metrics_df.columns
    ]
    if not cols:
        raise ValueError("No expected metric columns found for teacher/student chart")

    fig, axes = plt.subplots(1, len(cols), figsize=(4.8 * len(cols), 4), constrained_layout=True)
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        ax.bar(metrics_df["student"], metrics_df[col])
        ax.set_title(col.replace("_", " "))
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle(title)
    save_figure_pdf_png(fig, out_path_base)
    plt.close(fig)


def plot_training_curves(
    histories: dict,
    out_path_base: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, hist in histories.items():
        train = hist.get("train_loss_history", [])
        val = hist.get("val_loss_history", [])
        if train:
            ax.plot([x["step"] for x in train], [x["train_loss"] for x in train], label=f"{name}: train")
        if val:
            ax.plot([x["step"] for x in val], [x["val_loss"] for x in val], linestyle="--", label=f"{name}: val")

    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend(loc="best")
    save_figure_pdf_png(fig, out_path_base)
    plt.close(fig)


def plot_environment_summary_table(
    summary_df: pd.DataFrame,
    out_path_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, max(2.4, 0.65 * len(summary_df) + 1.5)))
    ax.axis("off")
    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)
    ax.set_title("Environment Summary", pad=14)
    save_figure_pdf_png(fig, out_path_base)
    plt.close(fig)


def plot_selected_decoding_example(
    example_dict: dict,
    tokenizer,
    out_path_base: Path,
) -> None:
    rows = example_dict.get("candidate_table", [])
    if not rows:
        raise ValueError("No candidate_table in example_dict")

    tokens = [
        tokenizer.sp.id_to_piece(int(r["first_token_id"])) if hasattr(tokenizer, "sp") else str(r["first_token_id"])
        for r in rows
    ]
    scores = [float(r.get("branch_score", 0.0)) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(tokens, scores)
    ax.set_title("Selected Decoding Candidate Scores")
    ax.set_xlabel("first token candidate")
    ax.set_ylabel("branch score")
    ax.tick_params(axis="x", rotation=25)
    save_figure_pdf_png(fig, out_path_base)
    plt.close(fig)


def save_line_plot(x, y, pdf_path: Path, png_path: Path, title: str, xlabel: str, ylabel: str, csv_path: Path | None = None) -> None:
    _ensure_parent(pdf_path)
    _ensure_parent(png_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    if csv_path is not None:
        _ensure_parent(csv_path)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([xlabel, ylabel])
            for xi, yi in zip(x, y):
                w.writerow([xi, yi])


def save_bar_plot(labels, values, pdf_path: Path, png_path: Path, title: str, ylabel: str, csv_path: Path | None = None) -> None:
    _ensure_parent(pdf_path)
    _ensure_parent(png_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    if csv_path is not None:
        _ensure_parent(csv_path)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["label", "value"])
            for l, v in zip(labels, values):
                w.writerow([l, v])
