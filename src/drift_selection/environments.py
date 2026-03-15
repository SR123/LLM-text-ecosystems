from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import random

import numpy as np

from .checkpoints import resume_generation_state, save_generation_state
from .selected_decoding import SelectedDecodingConfig, generate_selected_tokens
from .training import generate_neutral_tokens


@dataclass
class EnvironmentGenerationConfig:
    mode: str = "neutral"  # original | neutral | selected
    total_tokens: int = 100000
    prompt_len: int = 64
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = 0.95
    seed: int = 123
    selected_cfg: SelectedDecodingConfig | None = None
    max_new_tokens_per_prompt: int = 64
    resume: bool = True
    checkpoint_interval_prompts: int = 10


def _decode_if_possible(tokenizer, token_ids: list[int]) -> str:
    if tokenizer is None:
        return ""
    if hasattr(tokenizer, "sp"):
        return tokenizer.sp.decode(token_ids)
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids)
    return ""


def save_environment_tokens(
    out_dir: Path,
    env_name: str,
    token_ids: list[int],
    tokenizer,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = out_dir / f"{env_name}_token_ids.npy"
    txt_path = out_dir / f"{env_name}.txt"
    meta_path = out_dir / f"{env_name}_manifest.json"

    np.save(npy_path, np.asarray(token_ids, dtype=np.int32))
    decoded = _decode_if_possible(tokenizer, token_ids)
    txt_path.write_text(decoded, encoding="utf-8")

    meta = {
        "environment": env_name,
        "token_count": int(len(token_ids)),
        "token_ids_path": str(npy_path),
        "decoded_text_path": str(txt_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def load_environment_tokens(path: Path) -> list[int]:
    arr = np.load(path, allow_pickle=False)
    return [int(x) for x in arr.tolist()]


def environment_summary(token_ids: list[int], tokenizer) -> dict:
    unique = len(set(token_ids))
    total = len(token_ids)
    return {
        "tokens": total,
        "unique_tokens": unique,
        "type_token_ratio": unique / total if total else 0.0,
        "decoded_preview": _decode_if_possible(tokenizer, token_ids[:200]),
    }


def generate_original_environment_subset(
    train_ids: list[int],
    total_tokens: int,
    seed: int,
) -> list[int]:
    if not train_ids:
        return []
    total_tokens = min(int(total_tokens), len(train_ids))
    rng = random.Random(seed)
    if total_tokens >= len(train_ids):
        return list(train_ids)
    start = rng.randint(0, len(train_ids) - total_tokens)
    return list(train_ids[start : start + total_tokens])


def build_environment_from_prompts(
    model,
    prompt_bank: list[list[int]],
    cfg: EnvironmentGenerationConfig,
    out_dir: Path,
    device,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not prompt_bank:
        raise ValueError("prompt_bank is empty")

    rng = random.Random(cfg.seed)

    env_ids: list[int] = []
    prompt_index = 0

    state_path_dir = out_dir / "state"
    state_path_dir.mkdir(parents=True, exist_ok=True)
    if cfg.resume:
        state = resume_generation_state(state_path_dir)
        if state is not None:
            env_ids = [int(x) for x in state.get("env_ids", [])]
            prompt_index = int(state.get("prompt_index", 0))

    n_prompts_used = 0
    while len(env_ids) < cfg.total_tokens:
        prompt = prompt_bank[prompt_index % len(prompt_bank)]

        if cfg.mode == "selected":
            if cfg.selected_cfg is None:
                raise ValueError("selected_cfg is required for selected mode")
            generated = generate_selected_tokens(
                model=model,
                prompt_ids=prompt,
                max_new_tokens=cfg.max_new_tokens_per_prompt,
                cfg=cfg.selected_cfg,
                device=device,
                progress_bar=False,
            )
        elif cfg.mode == "neutral":
            generated = generate_neutral_tokens(
                model=model,
                prompt_ids=prompt,
                max_new_tokens=cfg.max_new_tokens_per_prompt,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported mode for prompt generation: {cfg.mode}")

        continuation = generated[len(prompt):]
        env_ids.extend(int(x) for x in continuation)
        prompt_index += 1
        n_prompts_used += 1

        if cfg.resume and n_prompts_used % max(1, cfg.checkpoint_interval_prompts) == 0:
            save_generation_state(
                state_path_dir,
                {
                    "prompt_index": prompt_index,
                    "env_ids": env_ids,
                    "mode": cfg.mode,
                },
            )

        # Randomly rotate prompt order slightly to avoid cycles.
        if prompt_index % len(prompt_bank) == 0:
            rng.shuffle(prompt_bank)

    env_ids = env_ids[: cfg.total_tokens]

    token_ids_path = out_dir / f"{cfg.mode}_token_ids.npy"
    np.save(token_ids_path, np.asarray(env_ids, dtype=np.int32))

    manifest = {
        "mode": cfg.mode,
        "token_count": len(env_ids),
        "token_ids_path": str(token_ids_path),
        "n_prompts_used": n_prompts_used,
        "config": asdict(cfg),
    }
    manifest_path = out_dir / f"{cfg.mode}_environment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
