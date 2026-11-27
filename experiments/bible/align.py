"""Align Jamaican NT verses with the World English Bible (WEB) by book:chapter:verse.

WEB comes from ebible.org (public domain), verse-per-line: "MAT 1:1 text".
Where the JNT merges verses ("span"), the WEB verses in that span are joined.
Output: data/derived/bible_pairs.jsonl, one record per aligned verse:
  {"ref", "book", "chapter", "verse", "jnt", "web"}
plus a short alignment report.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

NT = ["MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "1CO", "2CO", "GAL", "EPH", "PHP", "COL",
      "1TH", "2TH", "1TI", "2TI", "TIT", "PHM", "HEB", "JAS", "1PE", "2PE", "1JN", "2JN",
      "3JN", "JUD", "REV"]
# ebible.org VPL files use their own three-letter codes for some books.
VPL_CODES = {"MAR": "MRK", "JOH": "JHN", "PHI": "PHP", "JAM": "JAS",
             "1JO": "1JN", "2JO": "2JN", "3JO": "3JN"}
VPL_RE = re.compile(r"^([1-3A-Z]{3}) (\d+):(\d+) (.*)$")


def load_web(path: Path) -> dict[tuple[str, int, int], str]:
    web = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = VPL_RE.match(line.strip())
        code = VPL_CODES.get(m.group(1), m.group(1)) if m else None
        if code in NT:
            text = m.group(4).strip()
            if text:
                web[(code, int(m.group(2)), int(m.group(3)))] = text
    return web


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jnt", default="data/derived/jnt_verses.jsonl")
    ap.add_argument("--web", default="data/web/eng-web_vpl.txt")
    ap.add_argument("--out", default="data/derived/bible_pairs.jsonl")
    a = ap.parse_args()

    web = load_web(Path(a.web))
    jnt = [json.loads(l) for l in open(a.jnt, encoding="utf-8")]
    pairs, missing = [], []
    for v in jnt:
        lo, hi = v.get("span", [v["verse"], v["verse"]])
        parts = [web.get((v["book"], v["chapter"], k)) for k in range(lo, hi + 1)]
        if any(p is None for p in parts):
            missing.append(v["ref"])
            continue
        pairs.append({"ref": v["ref"], "book": v["book"], "chapter": v["chapter"],
                      "verse": v["verse"], "merged": hi > lo,
                      "jnt": v["text"], "web": " ".join(parts)})

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Report: WEB verses never covered by the JNT, per book.
    covered = set()
    for v in jnt:
        lo, hi = v.get("span", [v["verse"], v["verse"]])
        covered |= {(v["book"], v["chapter"], k) for k in range(lo, hi + 1)}
    uncovered = Counter(b for (b, c, k) in web if (b, c, k) not in covered)
    print(f"WEB NT verses: {len(web)}  JNT records: {len(jnt)}  aligned pairs: {len(pairs)}")
    print(f"JNT records with no WEB match: {len(missing)} {missing[:10]}")
    print("WEB verses without JNT counterpart, by book:", dict(uncovered) or "none")


if __name__ == "__main__":
    main()
