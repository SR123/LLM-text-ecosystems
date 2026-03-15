#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request
import urllib.error

CORPORA = {
    "jane_austen": [1342, 158, 161],
    "charles_darwin": [1228, 2009, 2300],
    "mark_twain": [74, 76, 1837],
    "mary_shelley": [84, 18247, 18246],
}


def gutenberg_url(book_id: int) -> str:
    return f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"


def download_one(url: str, dst: Path) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            text = resp.read()
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(text)
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download public-domain corpora where missing")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    raw_root = root / "GitHub" / "corpora" / "raw"

    for corpus, ids in CORPORA.items():
        target = raw_root / corpus
        target.mkdir(parents=True, exist_ok=True)
        for book_id in ids:
            dst = target / f"{book_id}.txt"
            if dst.exists():
                continue
            ok = download_one(gutenberg_url(book_id), dst)
            if ok:
                print(f"Downloaded {corpus}:{book_id}")
            else:
                print(f"Skipped {corpus}:{book_id} (network unavailable or source unreachable)")

    print("Corpus download step complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
