
from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


# ---------------------------------------------------------------------
# Paths / utilities
# ---------------------------------------------------------------------

def find_project_root(start: Optional[Path] = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if p.name == "Drift_and_selection":
            return p
        if (p / "GitHub").exists() and (p / "Nat_Paper").exists():
            return p
    return start


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def save_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------

@dataclass
class TaskConfig:
    train_min_digits: int = 2
    train_max_digits: int = 3
    test_min_digits: int = 4
    test_max_digits: int = 5
    n_train: int = 8000
    n_val: int = 1000
    n_test: int = 2000
    seed: int = 123


@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 256


@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 8
    clip_grad_norm: float = 1.0
    eval_every_epoch: bool = True
    seed: int = 123
    checkpoint_every_epoch: bool = True


# ---------------------------------------------------------------------
# Tokenization / vocab
# ---------------------------------------------------------------------

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<sep>"]


class SimpleTokenVocab:
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

    def encode(self, tokens: List[str], add_bos=False, add_eos=False) -> List[int]:
        out = []
        if add_bos:
            out.append(self.bos_id)
        out.extend(self.stoi[t] for t in tokens)
        if add_eos:
            out.append(self.eos_id)
        return out

    def decode(self, ids: List[int], stop_at_eos: bool = True) -> List[str]:
        toks = []
        for i in ids:
            tok = self.itos[i]
            if stop_at_eos and tok == "<eos>":
                break
            if tok == "<pad>":
                continue
            toks.append(tok)
        return toks

    def save(self, path: Path) -> None:
        ensure_dir(path.parent)
        path.write_text(json.dumps({"itos": self.itos}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SimpleTokenVocab":
        obj = json.loads(path.read_text(encoding="utf-8"))
        vocab = cls([])
        vocab.itos = obj["itos"]
        vocab.stoi = {tok: i for i, tok in enumerate(vocab.itos)}
        vocab.pad_id = vocab.stoi["<pad>"]
        vocab.bos_id = vocab.stoi["<bos>"]
        vocab.eos_id = vocab.stoi["<eos>"]
        vocab.sep_id = vocab.stoi["<sep>"]
        return vocab


def collect_vocab_tokens(examples: Iterable[dict]) -> List[str]:
    toks = []
    for ex in examples:
        toks.extend(ex["prompt_tokens"])
        toks.extend(ex["target_tokens"])
    return toks


# ---------------------------------------------------------------------
# Synthetic task generation
# ---------------------------------------------------------------------

DIGITS = [str(i) for i in range(10)]
PROCESS_MARKERS = {"WORK", "TRY", "FAIL", "REPAIR"}


def digits_of(n: int) -> List[str]:
    return list(str(n))


def int_from_digits(tokens: List[str]) -> int:
    return int("".join(tokens))


def sample_int_with_digits(min_digits: int, max_digits: int) -> int:
    n_digits = random.randint(min_digits, max_digits)
    first = random.randint(1, 9)
    rest = [str(random.randint(0, 9)) for _ in range(n_digits - 1)]
    return int(str(first) + "".join(rest))


def no_carry_sum_digits(a: int, b: int) -> List[str]:
    ad = digits_of(a)[::-1]
    bd = digits_of(b)[::-1]
    L = max(len(ad), len(bd))
    out = []
    for i in range(L):
        da = int(ad[i]) if i < len(ad) else 0
        db = int(bd[i]) if i < len(bd) else 0
        out.append(str((da + db) % 10))
    return out[::-1]


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
            trace.extend(["COL", str(i), str(da), str(db), "CIN", str(carry), "SUM", *list(str(s)), "W", str(write), "C", str(carry_out)])
        else:
            trace.extend(["COL", str(i), str(da), str(db), "SUM", *list(str(s)), "W", str(write), "C", str(carry_out)])
        out_digits.append(str(write))
        carry = carry_out
    if carry:
        out_digits.append(str(carry))
    result_digits = out_digits[::-1]
    return trace, result_digits


def carry_needed(a: int, b: int) -> bool:
    ad = digits_of(a)[::-1]
    bd = digits_of(b)[::-1]
    L = max(len(ad), len(bd))
    carry = 0
    for i in range(L):
        da = int(ad[i]) if i < len(ad) else 0
        db = int(bd[i]) if i < len(bd) else 0
        s = da + db + carry
        if s >= 10:
            return True
        carry = s // 10
    return False


def prompt_tokens_for_add(a: int, b: int, task_name: str) -> List[str]:
    return ["TASK", task_name, "A", *digits_of(a), "SEP", "B", *digits_of(b), "SOLVE"]


def make_addition_example(a: int, b: int, style: str = "process") -> dict:
    prompt = prompt_tokens_for_add(a, b, "ADD")
    trace, ans_digits = addition_steps(a, b)
    artifact = ["ANS", *ans_digits]
    if style == "process":
        target = [*trace, *artifact]
    elif style == "artifact":
        target = artifact
    else:
        raise ValueError(style)
    return {
        "task_family": "addition_process_vs_artifact",
        "style": style,
        "a": a,
        "b": b,
        "answer_digits": ans_digits,
        "prompt_tokens": prompt,
        "target_tokens": target,
    }


def make_search_repair_example(a: int, b: int, style: str = "full_search") -> dict:
    prompt = prompt_tokens_for_add(a, b, "SEARCH_REPAIR_ADD")
    trace, ans_digits = addition_steps(a, b)
    nc = no_carry_sum_digits(a, b)
    need_repair = carry_needed(a, b)

    if need_repair:
        full_search = ["TRY", "NOCARRY", *nc, "CHECK", "FAIL", "REPAIR", *trace, "ANS", *ans_digits]
        success_only = [*trace, "ANS", *ans_digits]
    else:
        full_search = ["TRY", "NOCARRY", *nc, "CHECK", "OK", "ANS", *ans_digits]
        success_only = ["DIRECT", *nc, "ANS", *ans_digits]

    artifact = ["ANS", *ans_digits]

    if style == "full_search":
        target = full_search
    elif style == "success_only":
        target = success_only
    elif style == "artifact":
        target = artifact
    else:
        raise ValueError(style)

    return {
        "task_family": "search_repair_addition",
        "style": style,
        "a": a,
        "b": b,
        "answer_digits": ans_digits,
        "carry_needed": need_repair,
        "prompt_tokens": prompt,
        "target_tokens": target,
    }


def _force_wrong_attempt(no_carry_digits: List[str], gold_digits: List[str]) -> List[str]:
    """Make sure the failed-attempt branch is actually wrong for every problem."""
    attempt = list(no_carry_digits)
    if attempt != gold_digits:
        return attempt
    if not attempt:
        return ["0"]
    last = int(attempt[-1])
    attempt[-1] = str((last + 1) % 10)
    return attempt


def make_environment_modes_example(a: int, b: int, style: str = "artifact_only") -> dict:
    prompt = prompt_tokens_for_add(a, b, "TASKB_ENV_ADD")
    trace, ans_digits = addition_steps(a, b)
    no_carry = no_carry_sum_digits(a, b)
    failed_attempt = _force_wrong_attempt(no_carry, ans_digits)
    need_carry = carry_needed(a, b)

    if style == "artifact_only":
        target = ["ANS", *ans_digits]
    elif style == "worked_trace":
        target = ["WORK", *trace, "ANS", *ans_digits]
    elif style == "failed_then_repair":
        target = ["TRY", "NOCARRY", *failed_attempt, "FAIL", "REPAIR", *trace, "ANS", *ans_digits]
    else:
        raise ValueError(style)

    return {
        "task_family": "taskB_environment_modes_v2",
        "style": style,
        "a": a,
        "b": b,
        "answer_digits": ans_digits,
        "carry_needed": need_carry,
        "failed_attempt_digits": failed_attempt,
        "prompt_tokens": prompt,
        "target_tokens": target,
    }


def unique_pairs(n: int, min_digits: int, max_digits: int, seed: int) -> List[Tuple[int, int]]:
    set_seed(seed)
    seen = set()
    pairs = []
    while len(pairs) < n:
        a = sample_int_with_digits(min_digits, max_digits)
        b = sample_int_with_digits(min_digits, max_digits)
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def build_task_family_dataset(task_family: str, styles: List[str], cfg: TaskConfig) -> Dict[str, Dict[str, List[dict]]]:
    out: Dict[str, Dict[str, List[dict]]] = {style: {} for style in styles}
    train_pairs = unique_pairs(cfg.n_train, cfg.train_min_digits, cfg.train_max_digits, cfg.seed)
    val_pairs = unique_pairs(cfg.n_val, cfg.train_min_digits, cfg.train_max_digits, cfg.seed + 1)
    test_pairs = unique_pairs(cfg.n_test, cfg.test_min_digits, cfg.test_max_digits, cfg.seed + 2)

    if task_family == "addition_process_vs_artifact":
        maker = make_addition_example
    elif task_family == "search_repair_addition":
        maker = make_search_repair_example
    elif task_family == "taskB_environment_modes_v2":
        maker = make_environment_modes_example
    else:
        raise ValueError(f"Unknown task_family={task_family!r}")

    for style in styles:
        train_examples = [maker(a, b, style=style) for a, b in train_pairs]
        val_examples = [maker(a, b, style=style) for a, b in val_pairs]
        test_examples = [maker(a, b, style=style) for a, b in test_pairs]
        for i, ex in enumerate(train_examples):
            ex["example_id"] = f"train_{i:06d}"
        for i, ex in enumerate(val_examples):
            ex["example_id"] = f"val_{i:06d}"
        for i, ex in enumerate(test_examples):
            ex["example_id"] = f"test_{i:06d}"
        out[style]["train"] = train_examples
        out[style]["val"] = val_examples
        out[style]["test"] = test_examples
    return out


def save_examples_jsonl(path: Path, examples: List[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def load_examples_jsonl(path: Path) -> List[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------
# Dataset + batching
# ---------------------------------------------------------------------

class CausalSeqDataset(torch.utils.data.Dataset):
    def __init__(self, examples: List[dict], vocab: SimpleTokenVocab):
        self.examples = examples
        self.vocab = vocab
        self.rows = []
        for ex in examples:
            prompt_ids = vocab.encode(ex["prompt_tokens"], add_bos=True, add_eos=False)
            target_ids = vocab.encode(ex["target_tokens"], add_bos=False, add_eos=True)
            full = prompt_ids + [vocab.sep_id] + target_ids
            labels = [-100] * (len(prompt_ids) + 1) + target_ids
            self.rows.append((full, labels, ex))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        full, labels, ex = self.rows[idx]
        return {"input_ids": full, "labels": labels, "meta": ex}


def collate_causal(batch, pad_id: int):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    labels = []
    for x in batch:
        inp = x["input_ids"] + [pad_id] * (max_len - len(x["input_ids"]))
        lab = x["labels"] + [-100] * (max_len - len(x["labels"]))
        input_ids.append(inp)
        labels.append(lab)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "meta": [x["meta"] for x in batch],
    }


# ---------------------------------------------------------------------
# Tiny causal transformer
# ---------------------------------------------------------------------

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
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyCausalTransformer(nn.Module):
    def __init__(self, vocab_size: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok(input_ids) + self.pos(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


# ---------------------------------------------------------------------
# Training / checkpointing
# ---------------------------------------------------------------------

def save_checkpoint(run_dir: Path, epoch: int, model, optimizer, history: List[dict], meta: dict):
    ensure_dir(run_dir)
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "history": history,
        "meta": meta,
    }
    torch.save(ckpt, run_dir / "state_latest.pt")
    save_json(run_dir / "checkpoint_state.json", {"last_completed_epoch": epoch, "meta": meta})


def load_checkpoint(run_dir: Path, model, optimizer=None):
    ckpt_path = run_dir / "state_latest.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optim_state"])
    return ckpt


def make_loader(examples: List[dict], vocab: SimpleTokenVocab, batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
    ds = CausalSeqDataset(examples, vocab)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_causal(b, vocab.pad_id),
    )


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    losses = []
    for batch in loader:
        inp = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(inp)
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def train_language_model(
    model: TinyCausalTransformer,
    train_examples: List[dict],
    val_examples: List[dict],
    vocab: SimpleTokenVocab,
    train_cfg: TrainConfig,
    run_dir: Path,
    resume: bool = True,
) -> dict:
    ensure_dir(run_dir)
    set_seed(train_cfg.seed)
    device = get_device()
    model = model.to(device)

    train_loader = make_loader(train_examples, vocab, train_cfg.batch_size, shuffle=True)
    val_loader = make_loader(val_examples, vocab, train_cfg.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    history: List[dict] = []
    start_epoch = 0

    if resume:
        ckpt = load_checkpoint(run_dir, model, optimizer)
        if ckpt is not None:
            history = ckpt.get("history", [])
            start_epoch = int(ckpt.get("epoch", 0))

    pbar = tqdm(range(start_epoch, train_cfg.epochs), desc=f"train:{run_dir.name}", unit="epoch")
    for epoch in pbar:
        model.train()
        batch_losses = []
        batch_bar = tqdm(train_loader, desc=f"epoch {epoch+1}", leave=False)
        for batch in batch_bar:
            inp = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(inp)
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)
            optimizer.step()
            batch_losses.append(loss.item())
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = float(np.mean(batch_losses))
        val_loss = evaluate_loss(model, val_loader, device) if train_cfg.eval_every_epoch else float("nan")
        rec = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        history.append(rec)
        pd.DataFrame(history).to_csv(run_dir / "metrics_history.csv", index=False)
        pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

        if train_cfg.checkpoint_every_epoch:
            save_checkpoint(run_dir, epoch + 1, model, optimizer, history, meta={"device": str(device)})

    return {
        "run_dir": str(run_dir),
        "device": str(device),
        "history": history,
    }


# ---------------------------------------------------------------------
# Generation / parsing / evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def generate_response(
    model: TinyCausalTransformer,
    vocab: SimpleTokenVocab,
    prompt_tokens: List[str],
    max_new_tokens: int = 128,
) -> List[str]:
    device = next(model.parameters()).device
    model.eval()
    ids = vocab.encode(prompt_tokens, add_bos=True, add_eos=False) + [vocab.sep_id]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        if x.size(1) > model.cfg.max_seq_len:
            x = x[:, -model.cfg.max_seq_len:]
        logits = model(x)
        next_id = int(torch.argmax(logits[0, -1]).item())
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == vocab.eos_id:
            break
    out = vocab.decode(x[0].tolist(), stop_at_eos=True)
    if "<sep>" in out:
        sep_idx = out.index("<sep>")
        return out[sep_idx + 1 :]
    return out


def extract_answer_digits(tokens: List[str]) -> Optional[str]:
    if "ANS" not in tokens:
        return None
    idx = tokens.index("ANS") + 1
    digits = []
    while idx < len(tokens) and tokens[idx].isdigit():
        digits.append(tokens[idx])
        idx += 1
    return "".join(digits) if digits else None


def normalize_digit_string(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    if not s.isdigit():
        return s
    # Compare numerically while preserving string outputs in saved rows.
    return str(int(s))


def extract_check_state(tokens: List[str]) -> Optional[str]:
    if "CHECK" not in tokens:
        return None
    idx = tokens.index("CHECK") + 1
    if idx < len(tokens) and tokens[idx] in {"OK", "FAIL"}:
        return tokens[idx]
    return None


def has_repair(tokens: List[str]) -> bool:
    return "REPAIR" in tokens


def extract_first_digit_span(tokens: List[str]) -> Optional[str]:
    idx = 0
    while idx < len(tokens):
        if tokens[idx].isdigit():
            digits = []
            while idx < len(tokens) and tokens[idx].isdigit():
                digits.append(tokens[idx])
                idx += 1
            return "".join(digits) if digits else None
        idx += 1
    return None


def extract_answer_digits_flexible(tokens: List[str]) -> Optional[str]:
    by_ans = extract_answer_digits(tokens)
    if by_ans is not None:
        return by_ans
    return extract_first_digit_span(tokens)


def first_process_marker_index(tokens: List[str]) -> Optional[int]:
    for i, tok in enumerate(tokens):
        if tok in PROCESS_MARKERS:
            return i
    return None


def contains_process_markers(tokens: List[str]) -> bool:
    return any(tok in PROCESS_MARKERS for tok in tokens)


def direct_clean_sequence_ok(tokens: List[str], gold_digits: str) -> bool:
    if contains_process_markers(tokens):
        return False
    if tokens == list(gold_digits):
        return True
    return tokens == ["ANS", *list(gold_digits)]


@torch.no_grad()
def sequence_logprob_after_prompt(
    model: TinyCausalTransformer,
    vocab: SimpleTokenVocab,
    prompt_tokens: List[str],
    target_tokens: List[str],
) -> float:
    """Teacher-forced log-probability of a target continuation right after prompt."""
    device = next(model.parameters()).device
    model.eval()
    ids = vocab.encode(prompt_tokens, add_bos=True, add_eos=False) + [vocab.sep_id]
    target_ids = vocab.encode(target_tokens, add_bos=False, add_eos=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    total = 0.0
    for tid in target_ids:
        if x.size(1) > model.cfg.max_seq_len:
            x = x[:, -model.cfg.max_seq_len:]
        logits = model(x)
        log_probs = torch.log_softmax(logits[0, -1], dim=-1)
        total += float(log_probs[tid].item())
        x = torch.cat([x, torch.tensor([[tid]], dtype=torch.long, device=device)], dim=1)
    return total


def is_structurally_valid_trace(tokens: List[str], expected_columns: int) -> bool:
    """Lightweight structural check for process-mode outputs."""
    if "ANS" not in tokens:
        return False
    col_count = sum(1 for tok in tokens if tok == "COL")
    return col_count >= max(1, expected_columns)


def repair_marker_structure_ok(tokens: List[str]) -> bool:
    has_try = "TRY" in tokens
    has_fail = "FAIL" in tokens
    has_repair_tok = "REPAIR" in tokens
    if has_fail != has_repair_tok:
        return False
    if has_fail:
        if not has_try:
            return False
        try_idx = tokens.index("TRY")
        fail_idx = tokens.index("FAIL")
        repair_idx = tokens.index("REPAIR")
        ans_idx = tokens.index("ANS") if "ANS" in tokens else len(tokens)
        return try_idx < fail_idx < repair_idx < ans_idx
    return True


@torch.no_grad()
def evaluate_direct_answer_mode(
    model: TinyCausalTransformer,
    vocab: SimpleTokenVocab,
    examples: List[dict],
    max_new_tokens: int = 48,
) -> dict:
    rows = []
    exact = 0
    in_prefix = 0
    clean = 0
    contaminated = 0
    logprobs = []

    for ex in tqdm(examples, desc="eval-direct-answer", leave=False):
        pred_tokens = generate_response(model, vocab, ex["prompt_tokens"], max_new_tokens=max_new_tokens)
        gold = "".join(ex["answer_digits"])
        pred = extract_answer_digits_flexible(pred_tokens)

        marker_idx = first_process_marker_index(pred_tokens)
        prefix_tokens = pred_tokens if marker_idx is None else pred_tokens[:marker_idx]
        prefix_pred = extract_answer_digits_flexible(prefix_tokens)
        has_markers = contains_process_markers(pred_tokens)

        exact_ok = normalize_digit_string(pred) == normalize_digit_string(gold)
        prefix_ok = normalize_digit_string(prefix_pred) == normalize_digit_string(gold)
        clean_ok = exact_ok and direct_clean_sequence_ok(pred_tokens, gold)

        exact += int(exact_ok)
        in_prefix += int(prefix_ok)
        clean += int(clean_ok)
        contaminated += int(has_markers)

        answer_lp = sequence_logprob_after_prompt(model, vocab, ex["prompt_tokens"], ["ANS", *list(gold)])
        logprobs.append(answer_lp)

        rows.append(
            {
                "example_id": ex.get("example_id"),
                "a": ex["a"],
                "b": ex["b"],
                "gold_answer": gold,
                "pred_answer": pred,
                "direct_exact_match": exact_ok,
                "direct_answer_in_prefix": prefix_ok,
                "direct_clean_answer": clean_ok,
                "contains_process_markers": has_markers,
                "direct_answer_logprob": answer_lp,
                "prediction_tokens": " ".join(pred_tokens),
            }
        )

    n = max(len(rows), 1)
    return {
        "direct_exact_match": exact / n,
        "direct_answer_in_prefix": in_prefix / n,
        "direct_clean_answer_rate": clean / n,
        "direct_marker_contamination_rate": contaminated / n,
        "direct_answer_logprob": float(np.mean(logprobs)) if logprobs else float("nan"),
        "rows": pd.DataFrame(rows),
    }


@torch.no_grad()
def evaluate_process_mode(
    model: TinyCausalTransformer,
    vocab: SimpleTokenVocab,
    examples: List[dict],
    max_new_tokens: int = 192,
) -> dict:
    rows = []
    final_correct = 0
    trace_valid = 0
    repair_acc = 0
    answer_present = 0

    for ex in tqdm(examples, desc="eval-process-mode", leave=False):
        process_prompt = [*ex["prompt_tokens"], "WORK"]
        pred_tokens = generate_response(model, vocab, process_prompt, max_new_tokens=max_new_tokens)
        gold = "".join(ex["answer_digits"])
        pred = extract_answer_digits_flexible(pred_tokens)
        expected_cols = max(len(str(ex["a"])), len(str(ex["b"])))

        final_ok = normalize_digit_string(pred) == normalize_digit_string(gold)
        trace_ok = is_structurally_valid_trace(pred_tokens, expected_columns=expected_cols) and final_ok
        repair_ok = repair_marker_structure_ok(pred_tokens)
        answer_in_completion = pred is not None

        final_correct += int(final_ok)
        trace_valid += int(trace_ok)
        repair_acc += int(repair_ok)
        answer_present += int(answer_in_completion)

        rows.append(
            {
                "example_id": ex.get("example_id"),
                "a": ex["a"],
                "b": ex["b"],
                "gold_answer": gold,
                "pred_answer": pred,
                "process_final_answer_accuracy": final_ok,
                "process_trace_validity": trace_ok,
                "repair_marker_correct": repair_ok,
                "process_answer_in_completion": answer_in_completion,
                "prediction_tokens": " ".join(pred_tokens),
            }
        )

    n = max(len(rows), 1)
    return {
        "process_final_answer_accuracy": final_correct / n,
        "process_trace_validity": trace_valid / n,
        "repair_marker_accuracy": repair_acc / n,
        "process_answer_in_completion": answer_present / n,
        "rows": pd.DataFrame(rows),
    }


def write_sample_generations_txt(path: Path, df: pd.DataFrame, n: int = 20) -> None:
    ensure_dir(path.parent)
    if df.empty:
        path.write_text("No generations available.\n", encoding="utf-8")
        return
    sample_df = df.head(n)
    lines = []
    for i, row in sample_df.iterrows():
        lines.append(f"Example {i + 1}")
        if "a" in row and "b" in row:
            lines.append(f"Problem: {row['a']} + {row['b']}")
        lines.append(f"Gold: {row.get('gold_answer', '')}")
        lines.append(f"Pred: {row.get('pred_answer', '')}")
        lines.append(f"Tokens: {row.get('prediction_tokens', '')}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


@torch.no_grad()
def evaluate_final_answer_accuracy(model, vocab, examples: List[dict], max_new_tokens: int = 128) -> dict:
    correct = 0
    strict_correct = 0
    total = 0
    rows = []
    for ex in tqdm(examples, desc="eval-final-answer", leave=False):
        pred_tokens = generate_response(model, vocab, ex["prompt_tokens"], max_new_tokens=max_new_tokens)
        pred = extract_answer_digits(pred_tokens)
        gold = "".join(ex["answer_digits"])
        ok = normalize_digit_string(pred) == normalize_digit_string(gold)
        strict_ok = pred == gold
        correct += int(ok)
        strict_correct += int(strict_ok)
        total += 1
        rows.append({
            "a": ex["a"],
            "b": ex["b"],
            "gold_answer": gold,
            "pred_answer": pred,
            "answer_correct": ok,
            "answer_correct_strict": strict_ok,
            "carry_needed": ex.get("carry_needed"),
            "prediction_tokens": " ".join(pred_tokens),
        })
    return {
        "answer_accuracy": correct / max(total, 1),
        "answer_accuracy_strict": strict_correct / max(total, 1),
        "rows": pd.DataFrame(rows),
    }


@torch.no_grad()
def evaluate_search_protocol(model, vocab, examples: List[dict], max_new_tokens: int = 192) -> dict:
    rows = []
    answer_correct = 0
    answer_correct_strict = 0
    protocol_correct = 0
    repair_correct = 0
    total = 0
    for ex in tqdm(examples, desc="eval-search-protocol", leave=False):
        pred_tokens = generate_response(model, vocab, ex["prompt_tokens"], max_new_tokens=max_new_tokens)
        pred_ans = extract_answer_digits(pred_tokens)
        gold_ans = "".join(ex["answer_digits"])
        ans_ok = normalize_digit_string(pred_ans) == normalize_digit_string(gold_ans)
        ans_ok_strict = pred_ans == gold_ans

        gold_check = "FAIL" if ex.get("carry_needed") else "OK"
        pred_check = extract_check_state(pred_tokens)
        check_ok = (pred_check == gold_check)

        gold_repair = bool(ex.get("carry_needed"))
        pred_repair = has_repair(pred_tokens)
        repair_ok = (pred_repair == gold_repair)

        answer_correct += int(ans_ok)
        answer_correct_strict += int(ans_ok_strict)
        protocol_correct += int(check_ok)
        repair_correct += int(repair_ok)
        total += 1
        rows.append({
            "a": ex["a"], "b": ex["b"], "carry_needed": ex.get("carry_needed"),
            "gold_answer": gold_ans, "pred_answer": pred_ans, "answer_correct": ans_ok,
            "answer_correct_strict": ans_ok_strict,
            "gold_check": gold_check, "pred_check": pred_check, "check_correct": check_ok,
            "gold_repair": gold_repair, "pred_repair": pred_repair, "repair_correct": repair_ok,
            "prediction_tokens": " ".join(pred_tokens),
        })

    return {
        "answer_accuracy": answer_correct / max(total, 1),
        "answer_accuracy_strict": answer_correct_strict / max(total, 1),
        "check_accuracy": protocol_correct / max(total, 1),
        "repair_accuracy": repair_correct / max(total, 1),
        "rows": pd.DataFrame(rows),
    }


def sample_generation_examples(model, vocab, examples: List[dict], n: int = 10, max_new_tokens: int = 192) -> pd.DataFrame:
    chosen = random.sample(examples, min(n, len(examples)))
    rows = []
    for ex in chosen:
        pred_tokens = generate_response(model, vocab, ex["prompt_tokens"], max_new_tokens=max_new_tokens)
        rows.append({
            "prompt": " ".join(ex["prompt_tokens"]),
            "gold": " ".join(ex["target_tokens"]),
            "pred": " ".join(pred_tokens),
            "carry_needed": ex.get("carry_needed"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------

def build_vocab_from_datasets(datasets_by_style: Dict[str, Dict[str, List[dict]]]) -> SimpleTokenVocab:
    toks = []
    for style, splits in datasets_by_style.items():
        toks.extend(collect_vocab_tokens(splits["train"]))
        toks.extend(collect_vocab_tokens(splits["val"]))
    return SimpleTokenVocab(toks)


def save_dataset_bundle(base_dir: Path, datasets_by_style: Dict[str, Dict[str, List[dict]]], cfg: TaskConfig) -> None:
    ensure_dir(base_dir)
    save_json(base_dir / "task_config.json", asdict(cfg))
    for style, splits in datasets_by_style.items():
        for split_name, examples in splits.items():
            save_examples_jsonl(base_dir / f"{style}_{split_name}.jsonl", examples)


def load_dataset_bundle(base_dir: Path, styles: List[str]) -> Dict[str, Dict[str, List[dict]]]:
    out = {}
    for style in styles:
        out[style] = {
            "train": load_examples_jsonl(base_dir / f"{style}_train.jsonl"),
            "val": load_examples_jsonl(base_dir / f"{style}_val.jsonl"),
            "test": load_examples_jsonl(base_dir / f"{style}_test.jsonl"),
        }
    return out


def build_problem_manifest_from_style(
    datasets_by_style: Dict[str, Dict[str, List[dict]]],
    reference_style: str = "artifact_only",
) -> dict:
    if reference_style not in datasets_by_style:
        raise ValueError(f"Missing reference style {reference_style!r} in datasets")
    manifest = {"reference_style": reference_style, "splits": {}}
    for split in ["train", "val", "test"]:
        rows = []
        for ex in datasets_by_style[reference_style][split]:
            rows.append(
                {
                    "example_id": ex.get("example_id"),
                    "a": ex["a"],
                    "b": ex["b"],
                    "answer": "".join(ex["answer_digits"]),
                    "carry_needed": bool(ex.get("carry_needed", False)),
                }
            )
        manifest["splits"][split] = rows
    return manifest


def train_style_models(
    task_family: str,
    datasets_by_style: Dict[str, Dict[str, List[dict]]],
    vocab: SimpleTokenVocab,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    out_root: Path,
) -> Dict[str, dict]:
    ensure_dir(out_root)
    results = {}
    for style, splits in datasets_by_style.items():
        run_dir = out_root / style
        model = TinyCausalTransformer(len(vocab.itos), model_cfg)
        info = train_language_model(model, splits["train"], splits["val"], vocab, train_cfg, run_dir, resume=True)
        ckpt = torch.load(run_dir / "state_latest.pt", map_location=get_device())
        model.load_state_dict(ckpt["model_state"])
        model = model.to(get_device())
        results[style] = {
            "model": model,
            "run_dir": str(run_dir),
            "history": info["history"],
        }
    return results


def evaluate_task_family(
    task_family: str,
    trained: Dict[str, dict],
    vocab: SimpleTokenVocab,
    datasets_by_style: Dict[str, Dict[str, List[dict]]],
    out_root: Path,
) -> pd.DataFrame:
    rows = []
    ensure_dir(out_root)
    for style, obj in trained.items():
        model = obj["model"]
        test_examples = datasets_by_style[style]["test"]
        if task_family == "addition_process_vs_artifact":
            ev = evaluate_final_answer_accuracy(model, vocab, test_examples)
            metrics = {
                "style": style,
                "task_family": task_family,
                "answer_accuracy": ev["answer_accuracy"],
                "answer_accuracy_strict": ev["answer_accuracy_strict"],
                "check_accuracy": np.nan,
                "repair_accuracy": np.nan,
            }
            ev["rows"].to_csv(out_root / f"{style}_test_predictions.csv", index=False)
        else:
            ev = evaluate_search_protocol(model, vocab, test_examples)
            metrics = {
                "style": style,
                "task_family": task_family,
                "answer_accuracy": ev["answer_accuracy"],
                "answer_accuracy_strict": ev["answer_accuracy_strict"],
                "check_accuracy": ev["check_accuracy"],
                "repair_accuracy": ev["repair_accuracy"],
            }
            ev["rows"].to_csv(out_root / f"{style}_test_predictions.csv", index=False)

        sample_df = sample_generation_examples(model, vocab, test_examples, n=12)
        sample_df.to_csv(out_root / f"{style}_samples.csv", index=False)

        rows.append(metrics)

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "summary_metrics.csv", index=False)
    return df


def evaluate_taskB_environment_modes(
    trained: Dict[str, dict],
    vocab: SimpleTokenVocab,
    datasets_by_style: Dict[str, Dict[str, List[dict]]],
    out_root: Path,
    reference_style: str = "artifact_only",
) -> dict:
    ensure_dir(out_root)
    direct_dir = ensure_dir(out_root / "evaluation_direct")
    process_dir = ensure_dir(out_root / "evaluation_process")
    figures_dir = ensure_dir(out_root / "figures")

    if reference_style not in datasets_by_style:
        raise ValueError(f"Missing reference style {reference_style!r} for shared evaluation")
    eval_examples = datasets_by_style[reference_style]["test"]

    direct_rows = []
    process_rows = []
    for style, obj in trained.items():
        model = obj["model"]

        direct_ev = evaluate_direct_answer_mode(model, vocab, eval_examples)
        direct_df = direct_ev["rows"]
        direct_df.to_csv(direct_dir / f"{style}_test_predictions.csv", index=False)
        write_sample_generations_txt(direct_dir / f"sample_generations_{style}.txt", direct_df, n=20)
        direct_rows.append(
            {
                "style": style,
                "task_family": "taskB_environment_modes_v2",
                "direct_exact_match": direct_ev["direct_exact_match"],
                "direct_answer_in_prefix": direct_ev["direct_answer_in_prefix"],
                "direct_clean_answer_rate": direct_ev["direct_clean_answer_rate"],
                "direct_marker_contamination_rate": direct_ev["direct_marker_contamination_rate"],
                "direct_answer_logprob": direct_ev["direct_answer_logprob"],
            }
        )

        process_ev = evaluate_process_mode(model, vocab, eval_examples)
        process_df = process_ev["rows"]
        process_df.to_csv(process_dir / f"{style}_test_predictions.csv", index=False)
        write_sample_generations_txt(process_dir / f"sample_generations_{style}.txt", process_df, n=20)
        process_rows.append(
            {
                "style": style,
                "task_family": "taskB_environment_modes_v2",
                "process_final_answer_accuracy": process_ev["process_final_answer_accuracy"],
                "process_trace_validity": process_ev["process_trace_validity"],
                "repair_marker_accuracy": process_ev["repair_marker_accuracy"],
                "process_answer_in_completion": process_ev["process_answer_in_completion"],
            }
        )

    direct_summary = pd.DataFrame(direct_rows)
    process_summary = pd.DataFrame(process_rows)
    direct_summary.to_csv(direct_dir / "summary_metrics.csv", index=False)
    process_summary.to_csv(process_dir / "summary_metrics.csv", index=False)

    combined = direct_summary.merge(process_summary, on=["style", "task_family"], how="inner")
    combined.to_csv(out_root / "summary_metrics.csv", index=False)

    return {
        "direct_summary": direct_summary,
        "process_summary": process_summary,
        "combined": combined,
        "direct_dir": direct_dir,
        "process_dir": process_dir,
        "figures_dir": figures_dir,
    }


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def save_plot(fig, out_path_base: Path):
    ensure_dir(out_path_base.parent)
    fig.savefig(str(out_path_base.with_suffix(".png")), dpi=180, bbox_inches="tight")
    fig.savefig(str(out_path_base.with_suffix(".pdf")), bbox_inches="tight")


def plot_task_family_metrics(df: pd.DataFrame, out_path_base: Path, title: str):
    metric_cols = [c for c in ["answer_accuracy", "check_accuracy", "repair_accuracy"] if c in df.columns and df[c].notna().any()]
    n = len(metric_cols)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, metric_cols):
        ax.bar(df["style"], df[col])
        ax.set_ylim(0, 1.0)
        ax.set_title(col.replace("_", " "))
        ax.set_ylabel("accuracy")
        ax.tick_params(axis='x', rotation=20)
    fig.suptitle(title)
    fig.tight_layout()
    save_plot(fig, out_path_base)
    plt.close(fig)


def plot_taskB_environment_mode_figures(
    direct_df: pd.DataFrame,
    process_df: pd.DataFrame,
    figures_dir: Path,
    run_name: str,
) -> Dict[str, Path]:
    ensure_dir(figures_dir)
    paths: Dict[str, Path] = {}

    # Figure B1: direct-answer metrics
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(direct_df["style"], direct_df["direct_exact_match"])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("direct_exact_match")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(direct_df["style"], direct_df["direct_clean_answer_rate"])
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("direct_clean_answer_rate")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle(f"{run_name}: Direct-answer mode")
    fig.tight_layout()
    out_b1 = figures_dir / "figure_B1_direct_answer_metrics"
    save_plot(fig, out_b1)
    plt.close(fig)
    paths["figure_B1"] = out_b1.with_suffix(".pdf")

    # Figure B2: process-mode metrics
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(process_df["style"], process_df["process_final_answer_accuracy"])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("process_final_answer_accuracy")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(process_df["style"], process_df["process_trace_validity"])
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("process_trace_validity")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle(f"{run_name}: Process mode")
    fig.tight_layout()
    out_b2 = figures_dir / "figure_B2_process_mode_metrics"
    save_plot(fig, out_b2)
    plt.close(fig)
    paths["figure_B2"] = out_b2.with_suffix(".pdf")

    # Figure B3: direct-mode contamination
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(direct_df["style"], direct_df["direct_marker_contamination_rate"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("rate")
    ax.set_title("Direct mode marker contamination")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out_b3 = figures_dir / "figure_B3_direct_mode_marker_contamination"
    save_plot(fig, out_b3)
    plt.close(fig)
    paths["figure_B3"] = out_b3.with_suffix(".pdf")

    return paths


def plot_training_histories(trained: Dict[str, dict], out_path_base: Path, title: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for style, obj in trained.items():
        hist = pd.DataFrame(obj["history"])
        if not hist.empty:
            ax.plot(hist["epoch"], hist["train_loss"], marker="o", label=f"{style} train")
            ax.plot(hist["epoch"], hist["val_loss"], marker="s", linestyle="--", label=f"{style} val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_plot(fig, out_path_base)
    plt.close(fig)


# ---------------------------------------------------------------------
# High-level workflow wrappers
# ---------------------------------------------------------------------

def default_output_roots(project_root: Optional[Path] = None) -> dict:
    root = find_project_root(project_root)
    gh = root / "GitHub"
    return {
        "data_out": ensure_dir(gh / "data" / "outputs" / "theorem2_process_learning"),
        "fig_out": ensure_dir(gh / "figures" / "appendix" / "theorem2_process_learning"),
        "nb_out": ensure_dir(gh / "notebooks" / "active"),
        "src_out": ensure_dir(gh / "src" / "drift_selection"),
    }


def run_task_pipeline(
    task_family: str,
    styles: List[str],
    task_cfg: TaskConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    run_name: str,
    project_root: Optional[Path] = None,
) -> dict:
    roots = default_output_roots(project_root)
    run_root = ensure_dir(roots["data_out"] / run_name)
    datasets_dir = ensure_dir(run_root / "datasets")
    models_dir = ensure_dir(run_root / "models")
    eval_dir = ensure_dir(run_root / "evaluation")

    datasets = build_task_family_dataset(task_family, styles, task_cfg)
    save_dataset_bundle(datasets_dir, datasets, task_cfg)

    vocab = build_vocab_from_datasets(datasets)
    vocab.save(run_root / "vocab.json")

    trained = train_style_models(task_family, datasets, vocab, model_cfg, train_cfg, models_dir)
    manifest = {
        "task_family": task_family,
        "styles": styles,
        "task_cfg": asdict(task_cfg),
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg),
        "run_name": run_name,
        "timestamp": timestamp(),
    }

    if task_family == "taskB_environment_modes_v2":
        eval_pack = evaluate_taskB_environment_modes(
            trained=trained,
            vocab=vocab,
            datasets_by_style=datasets,
            out_root=run_root,
            reference_style="artifact_only",
        )
        metrics_df = eval_pack["combined"]
        figure_paths = plot_taskB_environment_mode_figures(
            eval_pack["direct_summary"],
            eval_pack["process_summary"],
            eval_pack["figures_dir"],
            run_name=run_name,
        )

        # Also mirror key figures in the shared appendix figure folder.
        plot_training_histories(trained, roots["fig_out"] / f"{run_name}_training", title=f"{run_name} training")
        manifest["taskB_problem_manifest"] = str(datasets_dir / "taskB_problem_manifest.json")
        manifest["evaluation_direct_summary"] = str(eval_pack["direct_dir"] / "summary_metrics.csv")
        manifest["evaluation_process_summary"] = str(eval_pack["process_dir"] / "summary_metrics.csv")
        manifest["figure_paths"] = {k: str(v) for k, v in figure_paths.items()}
        save_json(datasets_dir / "taskB_problem_manifest.json", build_problem_manifest_from_style(datasets, reference_style="artifact_only"))
        save_json(run_root / "run_manifest.json", manifest)

        return {
            "run_root": run_root,
            "metrics_df": metrics_df,
            "direct_metrics_df": eval_pack["direct_summary"],
            "process_metrics_df": eval_pack["process_summary"],
            "datasets": datasets,
            "trained": trained,
            "vocab": vocab,
        }

    metrics_df = evaluate_task_family(task_family, trained, vocab, datasets, eval_dir)
    plot_task_family_metrics(metrics_df, roots["fig_out"] / f"{run_name}_metrics", title=run_name)
    plot_training_histories(trained, roots["fig_out"] / f"{run_name}_training", title=f"{run_name} training")
    save_json(run_root / "run_manifest.json", manifest)

    return {
        "run_root": run_root,
        "metrics_df": metrics_df,
        "datasets": datasets,
        "trained": trained,
        "vocab": vocab,
    }


def pretty_print_examples(examples: List[dict], n: int = 3) -> str:
    rows = []
    for ex in examples[:n]:
        rows.append("PROMPT: " + " ".join(ex["prompt_tokens"]))
        rows.append("TARGET: " + " ".join(ex["target_tokens"]))
        rows.append("")
    return "\n".join(rows)
