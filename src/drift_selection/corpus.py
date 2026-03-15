from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import random
import re


@dataclass
class CorpusRecord:
    name: str
    raw_dir: Path
    cleaned_dir: Path
    splits_dir: Path


def load_text_file(path: str | Path) -> str:
    """Read a UTF-8 text file safely."""
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def load_text_files(paths: list[str | Path]) -> str:
    """Concatenate multiple text files into one string."""
    return "\n\n".join(load_text_file(p) for p in paths)


def clean_literary_text(text: str) -> str:
    """Lightly normalize text while preserving punctuation and case."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def list_text_files(folder: Path) -> list[Path]:
    return sorted([p for p in folder.glob("*.txt") if p.is_file()])


def read_corpus_text(folder: Path, limit_files: int | None = None) -> str:
    files = list_text_files(folder)
    if limit_files is not None:
        files = files[:limit_files]
    chunks: list[str] = []
    for path in files:
        chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n\n".join(chunks)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_text(path: str | Path, text: str) -> None:
    """Write UTF-8 text safely."""
    write_text(Path(path), text)


def load_conan_doyle_split(base_dir: Path) -> dict[str, str]:
    """Load cleaned train/val/test split texts from base_dir."""
    out: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = base_dir / f"{split}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing Conan Doyle split file: {path}")
        out[split] = load_text_file(path)
    return out


def sample_prompt_text_windows(
    text: str,
    n_prompts: int,
    prompt_chars: int,
    seed: int,
) -> list[str]:
    """Sample random prompt snippets by character windows."""
    if not text or prompt_chars <= 0 or n_prompts <= 0:
        return []
    if len(text) <= prompt_chars:
        return [text[:prompt_chars]] * n_prompts
    rng = random.Random(seed)
    windows: list[str] = []
    max_start = len(text) - prompt_chars
    for _ in range(n_prompts):
        i = rng.randint(0, max_start)
        windows.append(text[i:i + prompt_chars])
    return windows


def sample_prompt_token_windows(
    token_ids: list[int],
    prompt_len: int,
    n_prompts: int,
    seed: int,
) -> list[list[int]]:
    """Sample random token windows for prompt banks."""
    if prompt_len <= 0 or n_prompts <= 0:
        return []
    if len(token_ids) <= prompt_len:
        return [list(token_ids[:prompt_len])] * n_prompts if token_ids else []
    rng = random.Random(seed)
    windows: list[list[int]] = []
    max_start = len(token_ids) - prompt_len
    for _ in range(n_prompts):
        start = rng.randint(0, max_start)
        windows.append(token_ids[start:start + prompt_len])
    return windows


def build_prompt_bank_from_split(
    token_ids: list[int],
    prompt_len: int = 64,
    n_prompts: int = 512,
    seed: int = 123,
) -> list[list[int]]:
    """Convenience wrapper for evaluation prompt bank generation."""
    return sample_prompt_token_windows(
        token_ids=token_ids,
        prompt_len=prompt_len,
        n_prompts=n_prompts,
        seed=seed,
    )


def train_val_test_split_words(text: str, seed: int = 13, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> tuple[str, str, str]:
    words = text.split()
    if not words:
        return "", "", ""
    rng = random.Random(seed)
    idx = list(range(len(words)))
    rng.shuffle(idx)

    n = len(words)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)

    train_idx = set(idx[:n_train])
    val_idx = set(idx[n_train:n_train + n_val])

    train_words, val_words, test_words = [], [], []
    for i, w in enumerate(words):
        if i in train_idx:
            train_words.append(w)
        elif i in val_idx:
            val_words.append(w)
        else:
            test_words.append(w)

    return " ".join(train_words), " ".join(val_words), " ".join(test_words)


def write_splits(split_dir: Path, train: str, val: str, test: str) -> dict[str, str]:
    split_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / "train.txt"
    val_path = split_dir / "val.txt"
    test_path = split_dir / "test.txt"
    train_path.write_text(train, encoding="utf-8")
    val_path.write_text(val, encoding="utf-8")
    test_path.write_text(test, encoding="utf-8")
    return {
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path),
    }


def iter_corpus_records(corpora_root: Path, names: Iterable[str]) -> list[CorpusRecord]:
    out = []
    for name in names:
        out.append(
            CorpusRecord(
                name=name,
                raw_dir=corpora_root / "raw" / name,
                cleaned_dir=corpora_root / "cleaned" / name,
                splits_dir=corpora_root / "splits" / name,
            )
        )
    return out
