from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def timestamp() -> str:
    """Return a stable UTC timestamp string."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def utc_now_iso() -> str:
    """Backward-compatible alias used by existing scripts."""
    return timestamp().replace("+00:00", "Z")


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_text(path: os.PathLike[str] | str, encoding: str = "utf-8") -> str:
    return Path(path).read_text(encoding=encoding)


def write_text(path: os.PathLike[str] | str, text: str, encoding: str = "utf-8") -> None:
    p = Path(path)
    if p.parent:
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding=encoding)


def write_json(path: os.PathLike[str] | str, payload: Any, indent: int = 2) -> None:
    p = Path(path)
    if p.parent:
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=indent, sort_keys=True) + "\n", encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load YAML files")
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    return obj or {}


def save_yaml(path: Path, obj: dict[str, Any]) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to write YAML files")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten_dict(value, full_key))
        else:
            out[full_key] = value
    return out


def format_runtime(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    mins, sec = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs}h {mins}m {sec}s"
    if mins > 0:
        return f"{mins}m {sec}s"
    return f"{sec}s"


def stable_slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def repo_root_from_script(script_path: os.PathLike[str] | str, root_arg: str | None = None) -> Path:
    if root_arg:
        return Path(root_arg).resolve()
    p = Path(script_path).resolve()
    return p.parents[2]
