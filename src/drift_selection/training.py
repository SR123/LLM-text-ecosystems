from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .transformer import TransformerConfig, build_tiny_gpt
from .utils import format_runtime


@dataclass
class TrainingConfig:
    batch_size: int = 32
    context_len: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 4000
    eval_interval: int = 200
    eval_batches: int = 50
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    clip_grad_norm: float = 1.0
    seed: int = 12345
    device_preference: str = "mps_or_cpu"
    early_stopping_patience: int = 6
    save_interval: int = 500


@dataclass
class TrainStats:
    loss: float
    steps: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_best_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_lm_batch(
    token_ids: list[int] | np.ndarray,
    batch_size: int,
    context_len: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    arr = np.asarray(token_ids, dtype=np.int64)
    if arr.size <= context_len + 1:
        raise ValueError("Not enough token ids to sample LM batch")

    starts = np.random.randint(0, arr.size - context_len - 1, size=batch_size)
    x = np.stack([arr[s : s + context_len] for s in starts], axis=0)
    y = np.stack([arr[s + 1 : s + context_len + 1] for s in starts], axis=0)

    x_t = torch.tensor(x, dtype=torch.long, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    return x_t, y_t


@torch.no_grad()
def estimate_validation_loss(
    model,
    token_ids,
    batch_size,
    context_len,
    eval_batches,
    device,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(eval_batches):
        xb, yb = sample_lm_batch(token_ids, batch_size=batch_size, context_len=context_len, device=device)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("inf")


def _ckpt_path(out_dir: Path, step: int) -> Path:
    return out_dir / "checkpoints" / f"ckpt_step_{step:07d}.pt"


def save_checkpoint(
    out_dir: Path,
    model,
    optimizer,
    step: int,
    train_loss: float,
    val_loss: float,
    config: dict,
) -> Path:
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = _ckpt_path(out_dir, step)
    state = {
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "config": config,
    }
    torch.save(state, path)
    return path


def load_latest_checkpoint(
    out_dir: Path,
    model,
    optimizer=None,
    map_location="cpu",
) -> dict | None:
    ckpt_dir = out_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    candidates = sorted(ckpt_dir.glob("ckpt_step_*.pt"))
    if not candidates:
        return None
    latest = candidates[-1]
    state = torch.load(latest, map_location=map_location)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    state["checkpoint_path"] = str(latest)
    return state


def _lr_for_step(step: int, cfg: TrainingConfig) -> float:
    if cfg.warmup_steps <= 0:
        return cfg.learning_rate
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    return cfg.learning_rate


def train_language_model(
    model,
    train_ids,
    val_ids,
    train_cfg: TrainingConfig,
    out_dir: Path,
    resume: bool = True,
    progress_bar: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(train_cfg.seed)
    device = get_best_device(prefer_mps=True)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    train_history: list[dict[str, float]] = []
    val_history: list[dict[str, float]] = []
    best_val = float("inf")
    best_ckpt: Path | None = None
    start_step = 0
    no_improve = 0

    if resume:
        state = load_latest_checkpoint(out_dir, model, optimizer=optimizer, map_location=device)
        if state is not None:
            start_step = int(state.get("step", 0)) + 1
            best_val = float(state.get("val_loss", best_val))

    t0 = time.time()
    iterator = range(start_step, train_cfg.max_steps)
    if progress_bar:
        iterator = tqdm(iterator, desc="train-lm", unit="step")

    final_ckpt: Path | None = None

    for step in iterator:
        model.train()
        for g in optimizer.param_groups:
            g["lr"] = _lr_for_step(step, train_cfg)

        accum_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for _ in range(max(1, train_cfg.gradient_accumulation_steps)):
            xb, yb = sample_lm_batch(
                train_ids,
                batch_size=train_cfg.batch_size,
                context_len=train_cfg.context_len,
                device=device,
            )
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            loss = loss / max(1, train_cfg.gradient_accumulation_steps)
            loss.backward()
            accum_loss += float(loss.item())

        if train_cfg.clip_grad_norm and train_cfg.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)
        optimizer.step()

        train_loss = accum_loss
        train_history.append({"step": float(step), "train_loss": float(train_loss)})

        if step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps - 1:
            val_loss = estimate_validation_loss(
                model,
                val_ids,
                batch_size=train_cfg.batch_size,
                context_len=train_cfg.context_len,
                eval_batches=train_cfg.eval_batches,
                device=device,
            )
            val_history.append({"step": float(step), "val_loss": float(val_loss)})

            ckpt_path = save_checkpoint(
                out_dir=out_dir,
                model=model,
                optimizer=optimizer,
                step=step,
                train_loss=float(train_loss),
                val_loss=float(val_loss),
                config=asdict(train_cfg),
            )
            final_ckpt = ckpt_path

            if val_loss < best_val:
                best_val = val_loss
                best_ckpt = ckpt_path
                no_improve = 0
            else:
                no_improve += 1

            if progress_bar:
                iterator.set_postfix({"train": f"{train_loss:.4f}", "val": f"{val_loss:.4f}"})

            if train_cfg.early_stopping_patience > 0 and no_improve >= train_cfg.early_stopping_patience:
                break

        elif step % train_cfg.save_interval == 0 and step > start_step:
            final_ckpt = save_checkpoint(
                out_dir=out_dir,
                model=model,
                optimizer=optimizer,
                step=step,
                train_loss=float(train_loss),
                val_loss=float("nan"),
                config=asdict(train_cfg),
            )

    runtime = time.time() - t0
    return {
        "final_checkpoint_path": str(final_ckpt) if final_ckpt else None,
        "best_checkpoint_path": str(best_ckpt) if best_ckpt else str(final_ckpt) if final_ckpt else None,
        "train_loss_history": train_history,
        "val_loss_history": val_history,
        "steps": len(train_history),
        "runtime_seconds": runtime,
        "runtime_human": format_runtime(runtime),
        "device": str(device),
    }


def load_trained_model(
    checkpoint_path: Path,
    config: TransformerConfig,
    device,
):
    model = build_tiny_gpt(config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def generate_neutral_tokens(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    device=None,
) -> list[int]:
    if device is None:
        device = next(model.parameters()).device
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    return [int(t) for t in out[0].tolist()]


def train_epoch(model, optimizer, batches, device: str = "cpu") -> TrainStats:
    model.train()
    total = 0.0
    steps = 0
    for x, y in batches:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        steps += 1
    return TrainStats(loss=total / max(steps, 1), steps=steps)


@torch.no_grad()
def eval_epoch(model, batches, device: str = "cpu") -> TrainStats:
    model.eval()
    total = 0.0
    steps = 0
    for x, y in batches:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total += float(loss.item())
        steps += 1
    return TrainStats(loss=total / max(steps, 1), steps=steps)
