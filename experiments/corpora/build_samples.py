"""Sentence samples and feature labels for the natural collections in data/.

For each collection (spoken and written kept apart where the collection has both), the
readers in tokenisation_profile.py give cleaned text. It is split into sentences
(spaCy), filtered (6-40 words, mostly letters), and a seeded sample of --n sentences
is drawn from across all files. Each sentence is written to
data/derived/sample_<collection>.jsonl in the same format as the Bible samples, so
extract.py runs on it unchanged.

Labels: plural, past and progressive from spaCy en_core_web_sm, with the same rules as
for the WEB (features.web_labels). This is a Standard-English tagger on non-standard
varieties: every label records source "spacy", and annotation sheets are written to
data/annotation/<collection>_<feature>.csv so its accuracy can be measured and the labels
corrected, as for the Bible pair. In LinCE (Hindi–English) only words the corpus tags as
English (lang1) are labelled; the language tags are kept per word for code-switching.

Variety-specific constructions that the standard tagger cannot label (AAE habitual
"be", copula absence) are collected as candidates in data/annotation/candidates_<collection>.csv
for hand annotation.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bible"))
sys.path.insert(0, str(HERE))

from common import DERIVED, FEATURES, words_with_spans, write_jsonl  # noqa: E402
from features import write_sheet, web_labels, is_word  # noqa: E402
import tokenisation_profile as tp  # noqa: E402

COLLECTIONS = {
    "ice-ja-written": lambda: tp.ice("ice-jamaica/CORPUS", "W"),
    "ice-ja-spoken": lambda: tp.ice("ice-jamaica/CORPUS", "S"),
    "aapb-howard": lambda: tp.aapb("storycorps"),
    "aapb-eotp": lambda: tp.aapb("aapb/eotp-interviews"),
    "ice-in-written": lambda: tp.ice("ice-india/Corpus", "W"),
    "ice-in-spoken": lambda: tp.ice("ice-india/Corpus", "S"),
    "wpa": lambda: tp.wpa(),
    "coraal": lambda: tp.coraal(),
    "lince": None,  # word lists with language tags, handled separately
}
SUBJ = r"(?:i|you|he|she|it|we|they|y'all|dey|dem)"
HABITUAL_BE = re.compile(rf"\b{SUBJ}\s+(?:always\s+|usually\s+|just\s+|still\s+)?be\s+\w+", re.I)
ZERO_COP = re.compile(rf"\b(?:he|she|they|we|you)\s+(?:not\s+)?(?:\w+ing|real|so|too|good|bad|"
                      rf"tired|gone|done|here|there|in|at|from)\b", re.I)


def clean(t):
    t = re.sub(r"\s+", " ", t).strip()
    return t


def good_sentence(s):
    ws = words_with_spans(s)
    n = len(ws)
    alpha = sum(1 for w, _, _ in ws if is_word(w))
    # leftover interviewer text transcribed in capitals
    caps = re.search(r"(?:\b[A-Z]{2,}\b\W+){3,}", s)
    return 6 <= n <= 40 and alpha / max(1, n) >= 0.7 and not caps


def sentences(nlp, texts, rng, max_words=120000):
    texts = [clean(t) for t in texts]
    texts = [t for t in texts if t]
    rng.shuffle(texts)
    chunks, n = [], 0
    for t in texts:
        # split long documents into paragraph-sized pieces for spaCy
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", t):
            if part:
                chunks.append(part)
                n += len(part.split())
        if n >= max_words:
            break
    out = []
    for doc in nlp.pipe(chunks, batch_size=64):
        for s in doc.sents:
            txt = s.text.strip()
            if good_sentence(txt):
                out.append(txt)
    return out


def lince_sentences():
    out = []
    for ws, ls in tp.lince():
        if not ls or any(l is None for l in ls):
            continue  # untagged test split
        txt = " ".join(ws)
        if good_sentence(txt):
            out.append((txt, [str(l) for l in ls]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="sentences per collection")
    ap.add_argument("--sheet-size", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", help="subset of collections")
    a = ap.parse_args()

    import spacy
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2_000_000
    keys = a.only or list(COLLECTIONS)
    for key in keys:
        rng = random.Random(a.seed)
        lids = None
        if key == "lince":
            pairs = lince_sentences()
            rng.shuffle(pairs)
            pairs = pairs[:a.n]
            sents = [t for t, _ in pairs]
            lids = [l for _, l in pairs]
        else:
            sents = sentences(nlp, list(COLLECTIONS[key]()), rng)
            rng.shuffle(sents)
            sents = sents[:a.n]
        rows, counts, cand = [], Counter(), []
        sheet_rows = {f: [] for f in FEATURES}
        for k, txt in enumerate(sents):
            spans = words_with_spans(txt)
            words = [w for w, _, _ in spans]
            lab = web_labels(nlp, txt, spans)
            if lids is not None:  # LinCE: words are the corpus tokens; keep English-tagged only
                if len(lids[k]) != len(words):
                    continue
                for f in FEATURES:
                    lab[f] = {i: y for i, y in lab[f].items() if lids[k][i] == "lang1"}
            ref = f"{key}:{k}"
            for f in FEATURES:
                for i, y in lab[f].items():
                    counts[(f, y)] += 1
                for i, w in enumerate(words):
                    if is_word(w):
                        ctx = " ".join(words[max(0, i - 6):i] + [f"[{w}]"] + words[i + 1:i + 7])
                        sheet_rows[f].append({"ref": ref, "word_idx": i, "word": w, "context": ctx,
                                              "auto": lab[f].get(i, "")})
            for rx, kind in ((HABITUAL_BE, "habitual_be"), (ZERO_COP, "copula_absence")):
                for m in rx.finditer(txt):
                    cand.append({"ref": ref, "kind": kind, "match": m.group(0), "sentence": txt,
                                 "gold": ""})
            row = {"ref": ref, "variety": key, "text": txt, "words": words,
                   "spans": [[s, e] for _, s, e in spans],
                   "labels": {f: {str(i): y for i, y in lab[f].items()} for f in FEATURES},
                   "sources": {f: {str(i): "spacy" for i in lab[f]} for f in FEATURES}}
            if lids is not None:
                row["lid"] = lids[k]
            rows.append(row)
        write_jsonl(DERIVED / f"sample_{key}.jsonl", rows)
        rng2 = random.Random(a.seed)
        for f in FEATURES:
            write_sheet(key, f, sheet_rows[f], a.sheet_size, rng2)
        if cand:
            p = Path("data/annotation") / f"candidates_{key}.csv"
            with open(p, "w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["ref", "kind", "match", "sentence", "gold"])
                w.writeheader()
                w.writerows(cand)
        print(f"{key:16s} {len(rows):4d} sentences | " +
              " ".join(f"{f} {counts[(f, 1)]}/{counts[(f, 0)]}" for f in FEATURES) +
              f" | candidates: {Counter(c['kind'] for c in cand)}")


if __name__ == "__main__":
    main()
