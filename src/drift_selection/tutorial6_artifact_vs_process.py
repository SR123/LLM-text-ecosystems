"""
Tutorial 6 experiment: artefact-learning vs process-learning in fixed text environments.

This module implements the experiments described in Tutorial 6 of the appendix.
The key design:
  - A single arithmetic task family (column addition, 2-4 digits).
  - Two fixed text environments constructed from the SAME underlying problems:
      * artefact-only: prompt -> ANS <digits>
      * worked-trace:  prompt -> WORK COL ... ANS <digits>
  - Multiple transformer architectures (1-layer, 2-layer, 4-layer) to show
    that the environment effect is robust across model capacity.
  - Two deployment modes (direct-answer, show-work) to separate what the
    environment teaches from what the learner is asked to emit.
  - An n-gram baseline to show that n-gram agents cannot exploit traces
    (they lack the capacity), motivating why transformers are needed here.

All figures are generated programmatically for the tutorial appendix.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


# =====================================================================
# Utilities
# =====================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# =====================================================================
# Task generation
# =====================================================================

DIGITS = [str(i) for i in range(10)]
PROCESS_MARKERS = {"WORK", "COL", "CIN", "SUM", "W", "C", "ANS"}


def digits_of(n: int) -> List[str]:
    return list(str(n))


def sample_int_with_digits(min_digits: int, max_digits: int) -> int:
    n_digits = random.randint(min_digits, max_digits)
    first = random.randint(1, 9)
    rest = [str(random.randint(0, 9)) for _ in range(n_digits - 1)]
    return int(str(first) + "".join(rest))


def addition_steps(a: int, b: int) -> Tuple[List[str], List[str]]:
    ad = digits_of(a)[::-1]
    bd = digits_of(b)[::-1]
    L = max(len(ad), len(bd))
    carry = 0
    out_digits = []
    trace = []
    for i in range(L):
        da = int(ad[i]) if i < len(ad) else 0
        db = int(bd[i]) if i < len(bd) else 0
        s = da + db + carry
        write = s % 10
        carry_out = s // 10
        if carry > 0:
            trace.extend(["COL", str(i), str(da), str(db), "CIN", str(carry),
                           "SUM", *list(str(s)), "W", str(write), "C", str(carry_out)])
        else:
            trace.extend(["COL", str(i), str(da), str(db),
                           "SUM", *list(str(s)), "W", str(write), "C", str(carry_out)])
        out_digits.append(str(write))
        carry = carry_out
    if carry:
        out_digits.append(str(carry))
    return trace, out_digits[::-1]


def carry_needed(a: int, b: int) -> bool:
    ad = digits_of(a)[::-1]
    bd = digits_of(b)[::-1]
    L = max(len(ad), len(bd))
    carry = 0
    for i in range(L):
        da = int(ad[i]) if i < len(ad) else 0
        db = int(bd[i]) if i < len(bd) else 0
        if da + db + carry >= 10:
            return True
        carry = (da + db + carry) // 10
    return False


def unique_pairs(n: int, min_digits: int, max_digits: int, seed: int) -> List[Tuple[int, int]]:
    set_seed(seed)
    seen = set()
    pairs = []
    while len(pairs) < n:
        a = sample_int_with_digits(min_digits, max_digits)
        b = sample_int_with_digits(min_digits, max_digits)
        if (a, b) not in seen:
            seen.add((a, b))
            pairs.append((a, b))
    return pairs


def make_example(a: int, b: int, style: str, prompt_mode: str = "DIRECT") -> dict:
    prompt = ["TASK", "ADD", "A", *digits_of(a), "SEP", "B", *digits_of(b), "MODE", prompt_mode]
    trace, ans_digits = addition_steps(a, b)
    if style == "artifact_only":
        target = ["ANS", *ans_digits]
    elif style == "worked_trace":
        target = ["WORK", *trace, "ANS", *ans_digits]
    else:
        raise ValueError(f"Unknown style: {style}")
    return {
        "style": style,
        "prompt_mode": prompt_mode,
        "a": a, "b": b,
        "answer_digits": ans_digits,
        "carry_needed": carry_needed(a, b),
        "prompt_tokens": prompt,
        "target_tokens": target,
    }


@dataclass
class ExperimentConfig:
    # Data
    train_min_digits: int = 2
    train_max_digits: int = 4
    test_min_digits: int = 2
    test_max_digits: int = 4
    n_train: int = 15000
    n_val: int = 2000
    n_test: int = 3000
    seed: int = 42
    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 12
    clip_grad_norm: float = 1.0
    # Model variants
    architectures: List[dict] = field(default_factory=lambda: [
        {"name": "1-layer", "n_layers": 1, "d_model": 128, "n_heads": 4, "d_ff": 512},
        {"name": "2-layer", "n_layers": 2, "d_model": 128, "n_heads": 4, "d_ff": 512},
        {"name": "4-layer", "n_layers": 4, "d_model": 128, "n_heads": 4, "d_ff": 512},
    ])
    styles: List[str] = field(default_factory=lambda: ["artifact_only", "worked_trace"])
    max_seq_len: int = 320
    dropout: float = 0.1


def build_datasets(cfg: ExperimentConfig) -> Dict[str, Dict[str, List[dict]]]:
    train_pairs = unique_pairs(cfg.n_train, cfg.train_min_digits, cfg.train_max_digits, cfg.seed)
    val_pairs = unique_pairs(cfg.n_val, cfg.train_min_digits, cfg.train_max_digits, cfg.seed + 1)
    test_pairs = unique_pairs(cfg.n_test, cfg.test_min_digits, cfg.test_max_digits, cfg.seed + 2)

    datasets = {}
    for style in cfg.styles:
        train_ex = []
        for a, b in train_pairs:
            # Include both prompt modes in training
            train_ex.append(make_example(a, b, style, "DIRECT"))
            train_ex.append(make_example(a, b, style, "WORK"))
        val_ex = []
        for a, b in val_pairs:
            val_ex.append(make_example(a, b, style, "DIRECT"))
            val_ex.append(make_example(a, b, style, "WORK"))
        test_direct = [make_example(a, b, style, "DIRECT") for a, b in test_pairs]
        test_work = [make_example(a, b, style, "WORK") for a, b in test_pairs]
        datasets[style] = {
            "train": train_ex,
            "val": val_ex,
            "test_direct": test_direct,
            "test_work": test_work,
        }
    return datasets


# =====================================================================
# Tokenization
# =====================================================================

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<sep>"]


class Vocab:
    def __init__(self, tokens: List[str]):
        uniq = []
        seen = set()
        for tok in SPECIAL_TOKENS + tokens:
            if tok not in seen:
                uniq.append(tok)
                seen.add(tok)
        self.itos = uniq
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.sep_id = self.stoi["<sep>"]

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens: List[str], add_bos=False, add_eos=False) -> List[int]:
        out = []
        if add_bos:
            out.append(self.bos_id)
        out.extend(self.stoi[t] for t in tokens)
        if add_eos:
            out.append(self.eos_id)
        return out

    def decode(self, ids: List[int], stop_at_eos=True) -> List[str]:
        toks = []
        for i in ids:
            tok = self.itos[i]
            if stop_at_eos and tok == "<eos>":
                break
            if tok == "<pad>":
                continue
            toks.append(tok)
        return toks


def build_vocab(datasets: dict) -> Vocab:
    all_tokens = set()
    for style_data in datasets.values():
        for split_data in style_data.values():
            for ex in split_data:
                all_tokens.update(ex["prompt_tokens"])
                all_tokens.update(ex["target_tokens"])
    return Vocab(sorted(all_tokens))


# =====================================================================
# Model
# =====================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        att = torch.softmax(scores, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok(input_ids) + self.pos(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# =====================================================================
# Dataset / batching
# =====================================================================

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, examples: List[dict], vocab: Vocab):
        self.rows = []
        for ex in examples:
            prompt_ids = vocab.encode(ex["prompt_tokens"], add_bos=True)
            target_ids = vocab.encode(ex["target_tokens"], add_eos=True)
            full = prompt_ids + [vocab.sep_id] + target_ids
            labels = [-100] * (len(prompt_ids) + 1) + target_ids
            self.rows.append((full, labels, ex))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate(batch, pad_id: int, max_len: int = 320):
    input_ids, labels = [], []
    metas = []
    for full, lab, ex in batch:
        full = full[:max_len]
        lab = lab[:max_len]
        pad_n = max_len - len(full)
        input_ids.append(full + [pad_id] * pad_n)
        labels.append(lab + [-100] * pad_n)
        metas.append(ex)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "meta": metas,
    }


# =====================================================================
# Training
# =====================================================================

def train_model(model, train_examples, val_examples, vocab, cfg: ExperimentConfig, device):
    from functools import partial
    ds_train = SeqDataset(train_examples, vocab)
    ds_val = SeqDataset(val_examples, vocab)
    loader_train = torch.utils.data.DataLoader(
        ds_train, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=partial(collate, pad_id=vocab.pad_id, max_len=cfg.max_seq_len))
    loader_val = torch.utils.data.DataLoader(
        ds_val, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=partial(collate, pad_id=vocab.pad_id, max_len=cfg.max_seq_len))

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for batch in loader_train:
            inp = batch["input_ids"].to(device)
            lab = batch["labels"].to(device)
            logits = model(inp)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                   lab[:, 1:].reshape(-1), ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            optimizer.step()
            losses.append(loss.item())
        train_loss = np.mean(losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in loader_val:
                inp = batch["input_ids"].to(device)
                lab = batch["labels"].to(device)
                logits = model(inp)
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                       lab[:, 1:].reshape(-1), ignore_index=-100)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        history.append({"epoch": epoch + 1, "train_loss": float(train_loss), "val_loss": float(val_loss)})
        print(f"  epoch {epoch+1}/{cfg.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    return history


# =====================================================================
# Generation and evaluation
# =====================================================================

@torch.no_grad()
def generate(model, vocab, prompt_tokens, max_new=128, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    ids = vocab.encode(prompt_tokens, add_bos=True) + [vocab.sep_id]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new):
        if x.size(1) > model.max_seq_len:
            x = x[:, -model.max_seq_len:]
        logits = model(x)
        next_id = int(torch.argmax(logits[0, -1]).item())
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == vocab.eos_id:
            break
    out = vocab.decode(x[0].tolist(), stop_at_eos=True)
    if "<sep>" in out:
        return out[out.index("<sep>") + 1:]
    return out


def extract_answer(tokens):
    if "ANS" not in tokens:
        return None
    idx = tokens.index("ANS") + 1
    digits = []
    while idx < len(tokens) and tokens[idx].isdigit():
        digits.append(tokens[idx])
        idx += 1
    return "".join(digits) if digits else None


def is_valid_trace(tokens, expected_cols):
    if "ANS" not in tokens:
        return False
    col_count = sum(1 for t in tokens if t == "COL")
    return col_count >= max(1, expected_cols)


def has_process_markers(tokens):
    markers = {"WORK", "COL", "CIN", "SUM"}
    return any(t in markers for t in tokens)


def evaluate_model(model, vocab, test_examples, mode="direct", device=None):
    """Evaluate a model on test examples in a given deployment mode.
    mode='direct': prompt with MODE DIRECT, measure clean answer accuracy
    mode='work': prompt with MODE WORK then WORK prefix, measure trace quality
    """
    results = []
    for ex in test_examples:
        gold = "".join(ex["answer_digits"])
        expected_cols = max(len(str(ex["a"])), len(str(ex["b"])))

        if mode == "direct":
            pred_tokens = generate(model, vocab, ex["prompt_tokens"], max_new=64, device=device)
            pred_ans = extract_answer(pred_tokens)
            norm_pred = str(int(pred_ans)) if pred_ans and pred_ans.isdigit() else pred_ans
            norm_gold = str(int(gold))
            exact = (norm_pred == norm_gold)
            clean = exact and not has_process_markers(pred_tokens)
            contaminated = has_process_markers(pred_tokens)
            results.append({
                "a": ex["a"], "b": ex["b"], "carry": ex["carry_needed"],
                "gold": gold, "pred": pred_ans,
                "exact": exact, "clean": clean, "contaminated": contaminated,
                "tokens": " ".join(pred_tokens),
            })
        elif mode == "work":
            work_prompt = [*ex["prompt_tokens"][:-1], "WORK"]  # replace MODE token
            # Actually, prompt_tokens ends with MODE WORK or MODE DIRECT
            # We want MODE WORK
            work_prompt = list(ex["prompt_tokens"])
            # Replace last token (the mode) with WORK
            work_prompt[-1] = "WORK"
            pred_tokens = generate(model, vocab, work_prompt, max_new=192, device=device)
            pred_ans = extract_answer(pred_tokens)
            norm_pred = str(int(pred_ans)) if pred_ans and pred_ans.isdigit() else pred_ans
            norm_gold = str(int(gold))
            ans_correct = (norm_pred == norm_gold)
            trace_valid = is_valid_trace(pred_tokens, expected_cols) and ans_correct
            results.append({
                "a": ex["a"], "b": ex["b"], "carry": ex["carry_needed"],
                "gold": gold, "pred": pred_ans,
                "ans_correct": ans_correct, "trace_valid": trace_valid,
                "tokens": " ".join(pred_tokens),
            })

    df = pd.DataFrame(results)
    if mode == "direct":
        return {
            "exact_match": df["exact"].mean(),
            "clean_answer": df["clean"].mean(),
            "contamination": df["contaminated"].mean(),
            "exact_carry": df[df["carry"]]["exact"].mean() if df["carry"].any() else 0.0,
            "exact_no_carry": df[~df["carry"]]["exact"].mean() if (~df["carry"]).any() else 0.0,
            "df": df,
        }
    else:
        return {
            "answer_accuracy": df["ans_correct"].mean(),
            "trace_validity": df["trace_valid"].mean(),
            "acc_carry": df[df["carry"]]["ans_correct"].mean() if df["carry"].any() else 0.0,
            "acc_no_carry": df[~df["carry"]]["ans_correct"].mean() if (~df["carry"]).any() else 0.0,
            "df": df,
        }


# =====================================================================
# N-gram baseline
# =====================================================================

def build_ngram_table(examples: List[dict], n: int = 3) -> dict:
    """Build an n-gram (character-level on tokens) table from training examples."""
    table = {}  # context -> Counter of next tokens
    for ex in examples:
        seq = ["<bos>"] + ex["prompt_tokens"] + ["<sep>"] + ex["target_tokens"] + ["<eos>"]
        for i in range(len(seq)):
            for order in range(1, n + 1):
                if i >= order:
                    ctx = tuple(seq[i - order:i])
                    nxt = seq[i]
                    if ctx not in table:
                        table[ctx] = Counter()
                    table[ctx][nxt] += 1
    return table


def ngram_generate(table: dict, prompt_tokens: List[str], n: int = 3, max_new: int = 64) -> List[str]:
    seq = ["<bos>"] + prompt_tokens + ["<sep>"]
    for _ in range(max_new):
        generated = None
        for order in range(n, 0, -1):
            ctx = tuple(seq[-order:])
            if ctx in table:
                counts = table[ctx]
                tokens, weights = zip(*counts.items())
                generated = random.choices(tokens, weights=weights, k=1)[0]
                break
        if generated is None:
            break
        if generated == "<eos>":
            break
        seq.append(generated)
    return seq[len(prompt_tokens) + 2:]  # skip <bos> prompt <sep>


def evaluate_ngram(table, test_examples, n=3, mode="direct", n_samples=5):
    """Evaluate n-gram model by majority vote over n_samples."""
    results = []
    for ex in test_examples:
        gold = "".join(ex["answer_digits"])
        if mode == "work":
            prompt = list(ex["prompt_tokens"])
            prompt[-1] = "WORK"
        else:
            prompt = ex["prompt_tokens"]

        preds = []
        for _ in range(n_samples):
            tokens = ngram_generate(table, prompt, n=n, max_new=128)
            preds.append(extract_answer(tokens))

        # majority vote
        valid_preds = [p for p in preds if p is not None]
        if valid_preds:
            pred = Counter(valid_preds).most_common(1)[0][0]
        else:
            pred = None

        norm_pred = str(int(pred)) if pred and pred.isdigit() else pred
        norm_gold = str(int(gold))
        exact = (norm_pred == norm_gold)
        results.append({"a": ex["a"], "b": ex["b"], "gold": gold, "pred": pred, "exact": exact})

    df = pd.DataFrame(results)
    return {"accuracy": df["exact"].mean(), "df": df}


# =====================================================================
# Figure generation
# =====================================================================

def _style_colors():
    return {"artifact_only": "#2196F3", "worked_trace": "#FF9800"}


def _arch_markers():
    return {"1-layer": "o", "2-layer": "s", "4-layer": "D"}


def plot_direct_vs_work(all_results: dict, save_path: Path):
    """Main figure: grouped bar chart showing direct-answer and show-work accuracy
    for each (architecture, environment) pair."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    colors = _style_colors()
    archs = sorted(set(k[0] for k in all_results.keys()))
    styles = ["artifact_only", "worked_trace"]
    style_labels = {"artifact_only": "Artefact-only", "worked_trace": "Worked-trace"}

    # Panel A: direct-answer mode
    ax = axes[0]
    x = np.arange(len(archs))
    w = 0.35
    for j, style in enumerate(styles):
        vals = [all_results.get((arch, style), {}).get("direct", {}).get("clean_answer", 0) for arch in archs]
        bars = ax.bar(x + j * w - w / 2, vals, w, label=style_labels[style], color=colors[style], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.1%}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(archs)
    ax.set_ylabel("Clean direct-answer accuracy")
    ax.set_title("(a) Direct-answer deployment", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left", fontsize=9)

    # Panel B: show-work mode
    ax = axes[1]
    for j, style in enumerate(styles):
        vals = [all_results.get((arch, style), {}).get("work", {}).get("trace_validity", 0) for arch in archs]
        bars = ax.bar(x + j * w - w / 2, vals, w, label=style_labels[style], color=colors[style], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.1%}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(archs)
    ax.set_ylabel("Trace validity (correct answer + valid trace)")
    ax.set_title("(b) Show-work deployment", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("Artefact-only vs worked-trace environments across architectures", fontsize=12, y=1.02)
    fig.tight_layout()
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_carry_breakdown(all_results: dict, save_path: Path):
    """Breakdown of direct-answer accuracy by carry/no-carry problems."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = _style_colors()
    archs = sorted(set(k[0] for k in all_results.keys()))
    styles = ["artifact_only", "worked_trace"]
    style_labels = {"artifact_only": "Artefact-only", "worked_trace": "Worked-trace"}

    for panel_idx, (carry_label, key) in enumerate([("No carry", "exact_no_carry"), ("Carry required", "exact_carry")]):
        ax = axes[panel_idx]
        x = np.arange(len(archs))
        w = 0.35
        for j, style in enumerate(styles):
            vals = [all_results.get((arch, style), {}).get("direct", {}).get(key, 0) for arch in archs]
            bars = ax.bar(x + j * w - w / 2, vals, w, label=style_labels[style], color=colors[style], alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.1%}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(archs)
        ax.set_ylabel("Direct-answer exact match")
        ax.set_title(f"({chr(97+panel_idx)}) {carry_label}", fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("Direct-answer accuracy split by carry requirement", fontsize=12, y=1.02)
    fig.tight_layout()
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_training_curves(all_histories: dict, save_path: Path):
    """Training loss curves for all (architecture, style) pairs."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    colors = _style_colors()
    markers = _arch_markers()
    style_labels = {"artifact_only": "Artefact-only", "worked_trace": "Worked-trace"}

    for panel_idx, loss_key in enumerate(["train_loss", "val_loss"]):
        ax = axes[panel_idx]
        for (arch, style), hist in sorted(all_histories.items()):
            epochs = [h["epoch"] for h in hist]
            vals = [h[loss_key] for h in hist]
            ax.plot(epochs, vals, marker=markers.get(arch, "o"), markersize=4,
                    color=colors[style], linestyle="-" if "artifact" in style else "--",
                    label=f"{arch} / {style_labels[style]}", alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy loss")
        ax.set_title(f"({'a' if panel_idx == 0 else 'b'}) {'Training' if panel_idx == 0 else 'Validation'} loss",
                      fontsize=11)
        if panel_idx == 0:
            ax.legend(fontsize=7, ncol=1, loc="upper right")

    fig.tight_layout()
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_ngram_comparison(ngram_results: dict, transformer_results: dict, save_path: Path):
    """Compare n-gram baseline with best transformer on direct-answer accuracy."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = []
    vals = []
    bar_colors = []

    # N-gram results
    for (style, mode), res in sorted(ngram_results.items()):
        sl = "Artefact" if "artifact" in style else "Worked-trace"
        ml = "direct" if mode == "direct" else "show-work"
        labels.append(f"Trigram\n{sl}\n({ml})")
        vals.append(res["accuracy"])
        bar_colors.append("#9E9E9E")

    # Best transformer (4-layer)
    for style in ["artifact_only", "worked_trace"]:
        for mode in ["direct", "work"]:
            key = ("4-layer", style)
            if key in transformer_results:
                res = transformer_results[key].get(mode, {})
                sl = "Artefact" if "artifact" in style else "Worked-trace"
                if mode == "direct":
                    val = res.get("clean_answer", 0)
                    ml = "direct"
                else:
                    val = res.get("trace_validity", 0)
                    ml = "show-work"
                labels.append(f"4-layer TF\n{sl}\n({ml})")
                vals.append(val)
                bar_colors.append(_style_colors()[style])

    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=bar_colors, alpha=0.85, width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.1%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Accuracy / trace validity")
    ax.set_title("N-gram baseline vs transformer: environment effect requires learner capacity", fontsize=11)
    ax.set_ylim(0, 1.15)

    fig.tight_layout()
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_sample_generations(all_results: dict, save_path: Path):
    """Show sample generations from the best model to illustrate what each environment produces."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))

    for panel_idx, (style, mode) in enumerate([("artifact_only", "direct"), ("worked_trace", "work")]):
        ax = axes[panel_idx]
        ax.axis("off")
        key = ("4-layer", style)
        if key not in all_results:
            continue
        df = all_results[key].get(mode, {}).get("df")
        if df is None or df.empty:
            continue

        # Pick 3 examples: one easy (no carry), one medium, one hard (carry)
        samples = []
        if "carry" in df.columns:
            easy = df[~df["carry"]].head(1)
            hard = df[df["carry"]].head(2)
            samples = pd.concat([easy, hard]).head(3)
        else:
            samples = df.head(3)

        style_label = "Artefact-only env → direct answer" if panel_idx == 0 else "Worked-trace env → show work"
        text_lines = [f"  {style_label}\n"]
        for _, row in samples.iterrows():
            text_lines.append(f"  {row['a']} + {row['b']} = {row['gold']}")
            tok_str = row.get("tokens", "")
            if len(tok_str) > 100:
                tok_str = tok_str[:100] + " ..."
            text_lines.append(f"    → {tok_str}")
            text_lines.append("")

        ax.text(0.02, 0.95, "\n".join(text_lines), transform=ax.transAxes,
                fontsize=9, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("Sample generations: what each environment teaches the learner", fontsize=12, y=1.01)
    fig.tight_layout()
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# =====================================================================
# Main experiment runner
# =====================================================================

def run_full_experiment(cfg: ExperimentConfig = None, output_dir: Path = None):
    if cfg is None:
        cfg = ExperimentConfig()
    if output_dir is None:
        output_dir = Path("tutorial6_outputs")
    ensure_dir(output_dir)
    fig_dir = output_dir / "figures"
    ensure_dir(fig_dir)

    device = get_device()
    print(f"Device: {device}")
    print(f"Building datasets: {cfg.n_train} train, {cfg.n_val} val, {cfg.n_test} test")
    print(f"Digit range: {cfg.train_min_digits}-{cfg.train_max_digits}")

    set_seed(cfg.seed)
    datasets = build_datasets(cfg)
    vocab = build_vocab(datasets)
    print(f"Vocab size: {len(vocab)}")

    # Save example previews
    previews = {}
    for style in cfg.styles:
        previews[style] = []
        for ex in datasets[style]["train"][:6]:
            previews[style].append({
                "prompt": " ".join(ex["prompt_tokens"]),
                "target": " ".join(ex["target_tokens"]),
            })
    with open(output_dir / "example_previews.json", "w") as f:
        json.dump(previews, f, indent=2)

    all_results = {}
    all_histories = {}

    # Train transformers
    for arch_spec in cfg.architectures:
        arch_name = arch_spec["name"]
        for style in cfg.styles:
            print(f"\n{'='*60}")
            print(f"Training: {arch_name} / {style}")
            print(f"{'='*60}")

            set_seed(cfg.seed)
            model = TinyTransformer(
                vocab_size=len(vocab),
                d_model=arch_spec["d_model"],
                n_heads=arch_spec["n_heads"],
                n_layers=arch_spec["n_layers"],
                d_ff=arch_spec["d_ff"],
                dropout=cfg.dropout,
                max_seq_len=cfg.max_seq_len,
            )
            print(f"  Parameters: {model.count_params():,}")

            history = train_model(model, datasets[style]["train"], datasets[style]["val"],
                                  vocab, cfg, device)
            all_histories[(arch_name, style)] = history

            # Evaluate in both modes
            print(f"  Evaluating direct-answer mode...")
            direct_res = evaluate_model(model, vocab, datasets[style]["test_direct"],
                                        mode="direct", device=device)
            print(f"    Clean answer: {direct_res['clean_answer']:.3f}")
            print(f"    Exact match:  {direct_res['exact_match']:.3f}")

            print(f"  Evaluating show-work mode...")
            work_res = evaluate_model(model, vocab, datasets[style]["test_work"],
                                      mode="work", device=device)
            print(f"    Answer accuracy: {work_res['answer_accuracy']:.3f}")
            print(f"    Trace validity:  {work_res['trace_validity']:.3f}")

            all_results[(arch_name, style)] = {"direct": direct_res, "work": work_res}

            # Save per-model results
            model_dir = output_dir / f"{arch_name}_{style}"
            ensure_dir(model_dir)
            direct_res["df"].to_csv(model_dir / "eval_direct.csv", index=False)
            work_res["df"].to_csv(model_dir / "eval_work.csv", index=False)
            pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)
            torch.save(model.state_dict(), model_dir / "model.pt")

    # N-gram baseline
    print(f"\n{'='*60}")
    print("N-gram baseline (trigram)")
    print(f"{'='*60}")
    ngram_results = {}
    for style in cfg.styles:
        print(f"  Building trigram table for {style}...")
        table = build_ngram_table(datasets[style]["train"], n=3)
        for mode in ["direct", "work"]:
            test_key = "test_direct" if mode == "direct" else "test_work"
            test_ex = datasets[style][test_key][:500]  # subsample for speed
            res = evaluate_ngram(table, test_ex, n=3, mode=mode, n_samples=5)
            ngram_results[(style, mode)] = res
            print(f"    {style}/{mode}: accuracy={res['accuracy']:.3f}")

    # Save summary table
    summary_rows = []
    for (arch, style), res in all_results.items():
        summary_rows.append({
            "architecture": arch,
            "environment": style,
            "direct_clean_answer": res["direct"]["clean_answer"],
            "direct_exact_match": res["direct"]["exact_match"],
            "direct_contamination": res["direct"]["contamination"],
            "direct_carry_acc": res["direct"]["exact_carry"],
            "direct_no_carry_acc": res["direct"]["exact_no_carry"],
            "work_answer_accuracy": res["work"]["answer_accuracy"],
            "work_trace_validity": res["work"]["trace_validity"],
            "work_carry_acc": res["work"]["acc_carry"],
            "work_no_carry_acc": res["work"]["acc_no_carry"],
        })
    for (style, mode), res in ngram_results.items():
        summary_rows.append({
            "architecture": "trigram",
            "environment": style,
            f"{'direct' if mode=='direct' else 'work'}_accuracy": res["accuracy"],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary_all_results.csv", index=False)
    print(f"\nSummary saved to {output_dir / 'summary_all_results.csv'}")

    # Generate figures
    print("\nGenerating figures...")
    plot_direct_vs_work(all_results, fig_dir / "figure_tutorial6_direct_vs_work.pdf")
    plot_carry_breakdown(all_results, fig_dir / "figure_tutorial6_carry_breakdown.pdf")
    plot_training_curves(all_histories, fig_dir / "figure_tutorial6_training_curves.pdf")
    plot_ngram_comparison(ngram_results, all_results, fig_dir / "figure_tutorial6_ngram_vs_transformer.pdf")
    plot_sample_generations(all_results, fig_dir / "figure_tutorial6_sample_generations.pdf")

    print("\nDone!")
    return {
        "results": all_results,
        "histories": all_histories,
        "ngram_results": ngram_results,
        "summary_df": summary_df,
        "vocab": vocab,
        "datasets": datasets,
    }


if __name__ == "__main__":
    run_full_experiment()
