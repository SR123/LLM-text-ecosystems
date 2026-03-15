from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_save_json(path: Path, obj: dict) -> None:
    _atomic_write(path, (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def atomic_save_numpy(path: Path, arr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.save(tmp, np.asarray(arr))
    # np.save appends .npy when missing; normalize path
    actual_tmp = tmp if tmp.exists() else Path(str(tmp) + ".npy")
    os.replace(actual_tmp, path)


def atomic_save_text(path: Path, text: str) -> None:
    _atomic_write(path, text.encode("utf-8"))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resume_generation_state(out_dir: Path) -> dict | None:
    state_path = out_dir / "generation_state.json"
    if not state_path.exists():
        return None
    return load_json(state_path)


def save_generation_state(out_dir: Path, state: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_json(out_dir / "generation_state.json", state)


def save_json_checkpoint(path: Path, payload: dict) -> None:
    atomic_save_json(path, payload)


def load_json_checkpoint(path: Path) -> dict:
    return load_json(path)


def save_pickle_checkpoint(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle_checkpoint(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def save_torch_checkpoint(path: Path, state: dict) -> None:
    if torch is None:
        raise RuntimeError("torch is not available")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_torch_checkpoint(path: Path) -> dict:
    if torch is None:
        raise RuntimeError("torch is not available")
    return torch.load(path, map_location="cpu")
