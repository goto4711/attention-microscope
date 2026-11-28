"""Shared helpers for the Bible parallel-pair pipeline."""
from __future__ import annotations

import json
import re
from pathlib import Path

# One word segmentation for both varieties: words keep internal hyphens and
# apostrophes ("a-go", "de-so", "don’t"); punctuation marks are their own words.
WORD_RE = re.compile(r"[\w’'-]+|[^\w\s]")

VARIETIES = ("web", "jnt")
FEATURES = ("plural", "past", "progressive")

DERIVED = Path("data/derived")
OUT = Path("experiments/out")


def words_with_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
