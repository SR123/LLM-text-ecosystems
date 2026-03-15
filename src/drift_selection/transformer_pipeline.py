from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import re

from .corpus import build_prompt_bank_from_split, load_conan_doyle_split, save_text
from .tokenization import (
    decode_ids,
    encode_split_dict,
    load_sentencepiece_tokenizer,
    load_token_ids,
    save_token_ids,
    tokenizer_special_tokens,
    train_sentencepiece_tokenizer,
)
from .utils import ensure_dir, load_yaml, timestamp


def resolve_path(root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return root / p


def load_main_config(root: Path, config_path: str) -> dict:
    cfg = load_yaml(resolve_path(root, config_path))
    return cfg


def load_selected_cfg(root: Path, path: str) -> dict:
    return load_yaml(resolve_path(root, path)).get("selected_decoding", {})


def load_tokenizer_cfg(root: Path, path: str) -> dict:
    return load_yaml(resolve_path(root, path)).get("tokenizer", {})


def _prepare_sentencepiece_training_text(text: str, max_line_chars: int = 2000) -> str:
    """
    Convert very long single-line corpora into line-broken text suitable for SentencePiece.

    SentencePiece skips lines longer than its default max_sentence_length (4192), so we
    proactively split text into manageable pseudo-sentences.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return ""

    # Split on sentence-like punctuation boundaries first.
    rough_sentences = re.split(r"(?<=[.!?;:])\s+", normalized)
    lines: list[str] = []
    for sent in rough_sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= max_line_chars:
            lines.append(sent)
            continue

        # Fallback chunking by words for unusually long spans without punctuation.
        words = sent.split()
        chunk: list[str] = []
        chunk_len = 0
        for w in words:
            w_len = len(w) + (1 if chunk else 0)
            if chunk and chunk_len + w_len > max_line_chars:
                lines.append(" ".join(chunk))
                chunk = [w]
                chunk_len = len(w)
            else:
                chunk.append(w)
                chunk_len += w_len
        if chunk:
            lines.append(" ".join(chunk))

    return "\n".join(lines) + "\n"


def ensure_tokenizer_and_encoded_splits(root: Path, main_cfg: dict, force: bool = False) -> dict:
    paths_cfg = main_cfg["paths"]
    corpus_cfg = main_cfg["corpus"]
    tokenizer_cfg_rel = main_cfg["configs"]["tokenizer_config_path"]

    processed_dir = ensure_dir(resolve_path(root, paths_cfg["processed_dir"]))
    split_dir = resolve_path(root, corpus_cfg["split_dir"])

    split_texts = load_conan_doyle_split(split_dir)
    tok_cfg = load_tokenizer_cfg(root, tokenizer_cfg_rel)

    tok_out = ensure_dir(resolve_path(root, tok_cfg["output_dir"]))
    tokenizer_model_path = tok_out / "conan_doyle_sp.model"

    train_text_path = tok_out / "train_text_for_sp.txt"
    sp_text = _prepare_sentencepiece_training_text(split_texts["train"])
    if not sp_text.strip():
        raise ValueError("Conan Doyle training split is empty after preprocessing for SentencePiece.")
    save_text(train_text_path, sp_text)

    if force or not tokenizer_model_path.exists():
        tokenizer_model_path = train_sentencepiece_tokenizer(
            input_text_path=train_text_path,
            output_dir=tok_out,
            vocab_size=int(tok_cfg.get("vocab_size", 1000)),
            model_type=str(tok_cfg.get("model_type", "bpe")),
            character_coverage=float(tok_cfg.get("character_coverage", 1.0)),
        )

    tokenizer = load_sentencepiece_tokenizer(tokenizer_model_path)
    encoded = encode_split_dict(tokenizer, split_texts)

    encoded_paths = {
        split: processed_dir / f"conan_doyle_{split}_ids.npy"
        for split in ("train", "val", "test")
    }

    for split, ids in encoded.items():
        if force or not encoded_paths[split].exists():
            save_token_ids(encoded_paths[split], ids)

    manifest = {
        "created_at": timestamp(),
        "tokenizer_model_path": str(tokenizer_model_path),
        "tokenizer_special_tokens": tokenizer_special_tokens(tokenizer),
        "vocab_size": int(tokenizer.vocab_size),
        "split_text_paths": {k: str((split_dir / f"{k}.txt")) for k in ("train", "val", "test")},
        "encoded_split_paths": {k: str(v) for k, v in encoded_paths.items()},
        "token_counts": {k: int(len(v)) for k, v in encoded.items()},
    }
    manifest_path = processed_dir / "tokenization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "tokenizer": tokenizer,
        "tokenizer_model_path": tokenizer_model_path,
        "encoded_paths": encoded_paths,
        "encoded_ids": {k: load_token_ids(v) for k, v in encoded_paths.items()},
        "manifest_path": manifest_path,
    }


def write_prompt_bank(
    out_dir: Path,
    tokenizer,
    token_ids: list[int],
    prompt_len: int,
    n_prompts: int,
    seed: int,
    name: str,
) -> Path:
    out_dir = ensure_dir(out_dir)
    prompts = build_prompt_bank_from_split(token_ids, prompt_len=prompt_len, n_prompts=n_prompts, seed=seed)

    path = out_dir / f"{name}_prompt_bank.json"
    preview = []
    for p in prompts[:20]:
        preview.append({"ids": p, "text": decode_ids(tokenizer, p)})

    payload = {
        "name": name,
        "prompt_len": int(prompt_len),
        "n_prompts": int(len(prompts)),
        "seed": int(seed),
        "prompts": prompts,
        "preview": preview,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_prompt_bank(path: Path) -> list[list[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [[int(x) for x in row] for row in data["prompts"]]
