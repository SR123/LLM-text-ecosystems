from __future__ import annotations

import re
from pathlib import Path


def strip_gutenberg_boilerplate(text: str) -> str:
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
    ]

    lower = text
    start_idx = 0
    for marker in start_markers:
        i = lower.find(marker)
        if i != -1:
            start_idx = i + len(marker)
            break

    end_idx = len(text)
    for marker in end_markers:
        i = lower.find(marker)
        if i != -1:
            end_idx = min(end_idx, i)
    return text[start_idx:end_idx]


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def clean_text(text: str) -> str:
    return normalize_text(strip_gutenberg_boilerplate(text))


def clean_file(src: Path, dst: Path) -> None:
    raw = src.read_text(encoding="utf-8", errors="ignore")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(clean_text(raw), encoding="utf-8")
