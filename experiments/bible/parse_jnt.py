"""Parse the Jamaican New Testament (text extracted from the BSWI PDF) into verses.

The PDF text has running page headers, bare page numbers, section headings with
cross-references, and verse numbers glued to the first word ("1Dis a di lis ...").
We walk the lines with a small state machine:

- "<Book> <n>" on its own line starts chapter n of that book;
- a line starting with the next expected verse number starts that verse;
- short, unpunctuated, capitalised lines followed by a cross-reference, a verse
  start or a chapter start are section headings and are dropped;
- everything else continues the current verse.

Output: JSONL with {"ref": "MAT 1:1", "book": "MAT", "chapter": 1, "verse": 1, "text": ...}.
Combined verses (e.g. "17-18") are kept as one record whose ref names the first verse
and with "span": [17, 18].
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# JNT book names in canonical order -> USFM codes (as used by the WEB VPL file).
BOOKS = [
    ("Matyu", "MAT"), ("Maak", "MRK"), ("Luuk", "LUK"), ("Jan", "JHN"), ("Aks", "ACT"),
    ("Ruoman", "ROM"), ("1 Korintiyan", "1CO"), ("2 Korintiyan", "2CO"),
    ("Gyalieshan", "GAL"), ("Ifiijan", "EPH"), ("Filipiyan", "PHP"), ("Koloshan", "COL"),
    ("1 Tesaluoniyan", "1TH"), ("2 Tesaluoniyan", "2TH"), ("1 Timoti", "1TI"),
    ("2 Timoti", "2TI"), ("Taitos", "TIT"), ("Failiiman", "PHM"), ("Iibruu", "HEB"),
    ("Jiemz", "JAS"), ("1 Piita", "1PE"), ("2 Piita", "2PE"), ("1 Jan", "1JN"),
    ("2 Jan", "2JN"), ("3 Jan", "3JN"), ("Juud", "JUD"), ("Revilieshan", "REV"),
]
NAME2CODE = dict(BOOKS)
CHAPTER_RE = re.compile(r"^(" + "|".join(re.escape(n) for n, _ in BOOKS) + r") (\d+)$")
VERSE_RE = re.compile(r"^(\d+)(?:-(\d+))?\s*(\S.*)$")
CROSSREF_RE = re.compile(r"^\(.*\d+\.\d+.*\)$")
HEADER_LINES = {
    "Di Jamiekan Nyuu Testiment",
    "© 2012 Bible Society of the West Indies.",
    "http://biblesocietywi.org/",
    "Jamaican Creole English - Bible - New Testament",
}
END_PUNCT = tuple(".,;:?!”’\"')—–-")


def clean_lines(path: Path) -> list[str]:
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line in HEADER_LINES or line.isdigit() or line in NAME2CODE:
            continue
        out.append(line)
    return out


def is_heading(line: str, nxt: str | None, expected: int) -> bool:
    if line.endswith(END_PUNCT) or not line[:1].isupper() or len(line.split()) > 10:
        return False
    if nxt is None:
        return False
    if CROSSREF_RE.match(nxt) or CHAPTER_RE.match(nxt):
        return True
    m = VERSE_RE.match(nxt)
    return bool(m and int(m.group(1)) == expected)


def parse(path: Path) -> list[dict]:
    lines = clean_lines(path)
    # Skip front matter: start at the first chapter line.
    start = next(i for i, l in enumerate(lines) if CHAPTER_RE.match(l))
    lines = lines[start:]
    verses: list[dict] = []
    book = chapter = None
    cur: dict | None = None
    last = 0

    def flush():
        nonlocal cur
        if cur is not None:
            cur["text"] = re.sub(r"\s+", " ", " ".join(cur.pop("_parts"))).strip()
            verses.append(cur)
            cur = None

    for i, line in enumerate(lines):
        if line.startswith("Esclarecimento"):  # Portuguese back matter of the PDF
            break
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        m = CHAPTER_RE.match(line)
        if m:
            flush()
            book, chapter, last = NAME2CODE[m.group(1)], int(m.group(2)), 0
            continue
        if CROSSREF_RE.match(line):
            continue
        m = VERSE_RE.match(line)
        # Accept the next verse, or a small jump (verses the translation merges or omits).
        if m and book and last < int(m.group(1)) <= last + 3:
            flush()
            v = int(m.group(1))
            v2 = int(m.group(2)) if m.group(2) else v
            cur = {"ref": f"{book} {chapter}:{v}", "book": book, "chapter": chapter,
                   "verse": v, "_parts": [m.group(3)]}
            if v2 != v:
                cur["span"] = [v, v2]
            last = v2
            continue
        if is_heading(line, nxt, last + 1):
            continue
        if cur is not None:
            cur["_parts"].append(line)
    flush()
    return verses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/patois/jnt.txt")
    ap.add_argument("--out", default="data/derived/jnt_verses.jsonl")
    a = ap.parse_args()
    verses = parse(Path(a.src))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for v in verses:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    print(f"{len(verses)} verses -> {a.out}")


if __name__ == "__main__":
    main()
