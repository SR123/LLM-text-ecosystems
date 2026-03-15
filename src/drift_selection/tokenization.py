from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np

try:
    import sentencepiece as spm
except Exception:  # pragma: no cover
    spm = None


@dataclass
class SentencePieceWrapper:
    model_path: Path
    sp: Any

    @property
    def vocab_size(self) -> int:
        return int(self.sp.get_piece_size())

    def bos_id(self) -> int:
        return int(self.sp.bos_id())

    def eos_id(self) -> int:
        return int(self.sp.eos_id())

    def unk_id(self) -> int:
        return int(self.sp.unk_id())

    def pad_id(self) -> int:
        return int(self.sp.pad_id())


def _require_sentencepiece() -> None:
    if spm is None:
        raise RuntimeError("sentencepiece is required; install with `pip install sentencepiece`")


def train_sentencepiece_tokenizer(
    input_text_path: str | Path,
    output_dir: str | Path,
    vocab_size: int = 1000,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
    user_defined_symbols: list[str] | None = None,
) -> Path:
    """Train a SentencePiece tokenizer and return the .model path."""
    _require_sentencepiece()
    input_text_path = Path(input_text_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = output_dir / "conan_doyle_sp"

    args = {
        "input": str(input_text_path),
        "model_prefix": str(model_prefix),
        "vocab_size": int(vocab_size),
        "model_type": model_type,
        "character_coverage": float(character_coverage),
        "unk_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "pad_id": 3,
        "hard_vocab_limit": False,
    }
    if user_defined_symbols:
        args["user_defined_symbols"] = ",".join(user_defined_symbols)

    spm.SentencePieceTrainer.train(**args)

    model_path = model_prefix.with_suffix(".model")
    manifest = {
        "input_text_path": str(input_text_path),
        "model_path": str(model_path),
        "vocab_size": int(vocab_size),
        "model_type": model_type,
        "character_coverage": float(character_coverage),
        "user_defined_symbols": user_defined_symbols or [],
    }
    (output_dir / "tokenizer_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return model_path


def load_sentencepiece_tokenizer(model_path: str | Path) -> SentencePieceWrapper:
    _require_sentencepiece()
    model_path = Path(model_path)
    sp = spm.SentencePieceProcessor()
    ok = sp.load(str(model_path))
    if not ok:
        raise RuntimeError(f"Failed to load SentencePiece model: {model_path}")
    return SentencePieceWrapper(model_path=model_path, sp=sp)


def encode_text(
    tokenizer: SentencePieceWrapper,
    text: str,
    add_bos: bool = False,
    add_eos: bool = False,
) -> list[int]:
    ids = list(tokenizer.sp.encode(text, out_type=int))
    if add_bos and tokenizer.bos_id() >= 0:
        ids = [tokenizer.bos_id()] + ids
    if add_eos and tokenizer.eos_id() >= 0:
        ids = ids + [tokenizer.eos_id()]
    return ids


def decode_ids(tokenizer: SentencePieceWrapper, ids: list[int]) -> str:
    if not ids:
        return ""
    return tokenizer.sp.decode(list(map(int, ids)))


def encode_split_dict(
    tokenizer: SentencePieceWrapper,
    split_texts: dict[str, str],
) -> dict[str, list[int]]:
    return {split: encode_text(tokenizer, text) for split, text in split_texts.items()}


def save_token_ids(path: str | Path, token_ids: list[int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(token_ids, dtype=np.int32)
    np.save(path, arr)


def load_token_ids(path: str | Path) -> list[int]:
    arr = np.load(Path(path), allow_pickle=False)
    return [int(x) for x in arr.tolist()]


def tokenizer_special_tokens(tokenizer: SentencePieceWrapper) -> dict[str, int]:
    return {
        "bos": tokenizer.bos_id(),
        "eos": tokenizer.eos_id(),
        "unk": tokenizer.unk_id(),
        "pad": tokenizer.pad_id(),
    }


def word_tokenize(text: str) -> list[str]:
    return text.split()


def char_tokenize(text: str) -> list[str]:
    return list(text)
