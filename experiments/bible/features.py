"""Step 2: word-level feature labels for the Bible pair.

Three features that mean the same in both varieties but are marked differently:

  plural       WEB: noun inflection (-s)            JNT: postposed "dem" (di man dem)
  past         WEB: verb inflection (-ed, irregular) JNT: preverbal did / en / ben / neehn
  progressive  WEB: be + V-ing                       JNT: preverbal a (im a go)

Labels are 1 (feature present), 0 (a comparable word without it), or absent
(the word is not a candidate). Sources:

- WEB: spaCy en_core_web_sm (a Standard-English tagger; its accuracy on this text
  must be reported separately, see the WEB annotation sheet).
- JNT: transparent rules on the Creole markers below. They only label words where
  the marking is overt, so they *under*-cover JNT: bare plurals and unmarked past
  are left unlabelled. That is exactly what the hand annotation must add.

Hand corrections: the script writes annotation sheets (CSV) to data/annotation/.
Fill the `gold` column (1, 0, or x = not a candidate) and re-run;
for plural, label the word that ends the noun phrase (JNT "dem", or the noun
itself when no "dem" follows); gold labels
override rule/tagger labels and every label records its source.

Also selects the verse sample used by extract.py and probe.py, so that every step
uses the same verses and the same word segmentation.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from common import DERIVED, FEATURES, read_jsonl, words_with_spans, write_jsonl

# --- JNT rules -------------------------------------------------------------
PAST_MARKERS = {"did", "en", "ben", "neehn"}
# future / modal markers; "fi" (infinitive) and serial "go" are left out on purpose
NONPAST_MARKERS = {"wi", "a-go", "kyan", "kyaan", "mos", "shuda", "wuda", "kuda"}
SUBJECTS = {"mi", "yu", "im", "shi", "it", "wi", "unu", "dem", "aal"}
DETERMINERS = {"di", "dem", "aal", "som", "nof", "uol", "plenti", "mi", "yu", "im", "ar",
               "wi", "unu", "fi", "dis", "dat", "dem-de", "ada", "tuu", "chrii", "fuor"}
SINGULAR_DETS = {"wan", "dis", "da", "dat", "iich", "evri"}
# Words that never fill the verb or noun slot after a marker.
FUNCTION = {"a", "de", "di", "dem", "an", "se", "fi", "we", "wa", "so", "no", "iina", "pan",
            "wid", "tu", "fram", "bout", "laik", "kaaz", "kaa", "bot", "ef", "wen", "ou", "uu",
            "dat", "dis", "da", "ya", "mi", "yu", "im", "shi", "it", "wi", "unu", "ar", "wan",
            "aal", "nof", "som", "stil", "jos", "aalwiez", "nevaa", "neva", "ongl", "an", "op",
            "dong", "out", "iivn", "agen", "tuu", "wel", "nou", "den", "afta", "bifuo", "til",
            "did", "en", "ben", "neehn", "wi", "a-go", "kyan", "kyaan", "mos", "afi", "don"}


def is_word(w: str) -> bool:
    return any(ch.isalpha() for ch in w)


def jnt_labels(words: list[str]) -> dict[str, dict[int, int]]:
    lw = [w.lower() for w in words]
    lab = {f: {} for f in FEATURES}
    n = len(lw)
    for i, w in enumerate(lw):
        if not is_word(w):
            continue
        # plural: X dem, where X follows a determiner/possessor within two words.
        # The label sits on the NP-final word ("dem"), not on X: Pythia is causal,
        # so X cannot yet see the "dem" that marks it. WEB plurals likewise sit on
        # the NP-final noun that carries -s.
        if i + 1 < n and lw[i + 1] == "dem" and w not in FUNCTION:
            left = lw[max(0, i - 2):i]
            poss = any(words[j][:1].isupper() and j > 0 for j in range(max(0, i - 2), i))
            if any(x in DETERMINERS for x in left) or poss or words[i][:1].isupper():
                lab["plural"][i + 1] = 1
        # singular: wan / dis / da / iich X (not followed by dem)
        if i >= 1 and lw[i - 1] in SINGULAR_DETS and w not in FUNCTION and \
                not (i + 1 < n and lw[i + 1] == "dem"):
            lab["plural"][i] = 0
        # verb slot after a TMA marker (skip one optional "no"/"jos"/"stil")
        j = i - 1
        if j >= 0 and lw[j] in {"no", "jos", "stil", "aalwiez"}:
            j -= 1
        if j < 0 or w in FUNCTION or words[i][:1].isupper():
            continue
        m = lw[j]
        if m in PAST_MARKERS:
            lab["past"][i] = 1
            lab["progressive"][i] = 0
        elif m in NONPAST_MARKERS:
            lab["past"][i] = 0
            lab["progressive"][i] = 0
        elif m == "a" and j >= 1 and lw[j - 1] in SUBJECTS | PAST_MARKERS:
            # "im a go" / "dem did a tingk": progressive; "did a" is also past
            lab["progressive"][i] = 1
            lab["past"][i] = 1 if lw[j - 1] in PAST_MARKERS else 0
    return lab


# --- WEB via spaCy ----------------------------------------------------------
def web_labels(nlp, text: str, spans) -> dict[str, dict[int, int]]:
    doc = nlp(text)
    start2tok = {t.idx: t for t in doc}
    lab = {f: {} for f in FEATURES}
    for i, (w, s, e) in enumerate(spans):
        t = start2tok.get(s)
        if t is None or not is_word(w):
            continue
        if t.pos_ in {"NOUN", "PROPN"} and t.tag_ in {"NN", "NNS"}:
            lab["plural"][i] = int(t.tag_ == "NNS")
        if t.pos_ == "VERB":
            if t.tag_ == "VBD":
                lab["past"][i] = 1
            elif t.tag_ in {"VBP", "VBZ"} or (t.tag_ == "VB" and any(
                    c.dep_ == "aux" and c.tag_ == "MD" for c in t.children)):
                lab["past"][i] = 0
            aux_be = [c for c in t.children if c.dep_ == "aux" and c.lemma_ == "be"]
            if t.tag_ == "VBG" and aux_be:
                lab["progressive"][i] = 1
                lab["past"][i] = int(aux_be[0].tag_ == "VBD")
            elif t.tag_ in {"VBD", "VBP", "VBZ"}:
                lab["progressive"][i] = 0
    return lab


# --- annotation sheets ------------------------------------------------------
def sheet_path(variety, feature):
    return Path("data/annotation") / f"{variety}_{feature}.csv"


def load_gold(variety, feature) -> dict[tuple[str, int], str]:
    p = sheet_path(variety, feature)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        gold = {(r["ref"], int(r["word_idx"])): r["gold"].strip().lower()
                for r in csv.DictReader(f) if r.get("gold", "").strip()}
    return {**gold, **load_recode(variety, feature)}


def load_recode(variety, feature) -> dict[tuple[str, int], str]:
    """Reviewed changes to hand labels (data/annotation/<variety>_recode.csv), e.g. after a
    second annotator: applied on top of the gold column, which itself stays as annotated."""
    p = Path("data/annotation") / f"{variety}_recode.csv"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return {(r["ref"], int(r["word_idx"])): r["gold"].strip()
                for r in csv.DictReader(f) if r["feature"] == feature}


def write_sheet(variety, feature, rows, k, rng):
    """Sample up to k candidates (balanced 1/0) plus k//2 unlabelled content words."""
    p = sheet_path(variety, feature)
    if p.exists():
        print(f"  keep existing sheet {p} (delete it to resample)")
        return
    pos = [r for r in rows if r["auto"] == 1]
    neg = [r for r in rows if r["auto"] == 0]
    unl = [r for r in rows if r["auto"] == ""]
    pick = rng.sample(pos, min(len(pos), k // 2)) + rng.sample(neg, min(len(neg), k // 2)) \
        + rng.sample(unl, min(len(unl), k // 2))
    rng.shuffle(pick)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ref", "word_idx", "word", "context", "auto", "gold"])
        w.writeheader()
        for r in pick:
            w.writerow({**r, "gold": ""})
    print(f"  wrote {p} ({len(pick)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(DERIVED / "bible_pairs.jsonl"))
    ap.add_argument("--n", type=int, default=1000, help="verses in the sample")
    ap.add_argument("--max-words", type=int, default=60)
    ap.add_argument("--ratio", type=float, default=2.0,
                    help="drop pairs whose word-count ratio exceeds this (loose translations)")
    ap.add_argument("--sheet-size", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import spacy
    nlp = spacy.load("en_core_web_sm")
    rng = random.Random(a.seed)

    pairs = read_jsonl(a.pairs)
    keep = []
    for p in pairs:
        nj, nw = len(words_with_spans(p["jnt"])), len(words_with_spans(p["web"]))
        if max(nj, nw) <= a.max_words and max(nj, nw) / max(1, min(nj, nw)) <= a.ratio:
            keep.append(p)
    sample = sorted(rng.sample(keep, min(a.n, len(keep))), key=lambda p: pairs.index(p))
    print(f"{len(keep)} of {len(pairs)} pairs pass the length filters; sampled {len(sample)}")

    for variety in ("web", "jnt"):
        gold = {f: load_gold(variety, f) for f in FEATURES}
        rows, sheet_rows = [], {f: [] for f in FEATURES}
        counts = Counter()
        for p in sample:
            spans = words_with_spans(p[variety])
            words = [w for w, _, _ in spans]
            auto = web_labels(nlp, p[variety], spans) if variety == "web" else jnt_labels(words)
            labels, sources = {}, {}
            for f in FEATURES:
                lf, sf = {}, {}
                for i, w in enumerate(words):
                    g = gold[f].get((p["ref"], i))
                    if g in {"0", "1"}:
                        lf[i], sf[i] = int(g), "gold"
                    elif g == "x":
                        continue
                    elif i in auto[f]:
                        lf[i], sf[i] = auto[f][i], "spacy" if variety == "web" else "rule"
                    if (is_word(w) and w.lower() not in FUNCTION) or i in auto[f]:
                        ctx = " ".join(words[max(0, i - 6):i] + [f"[{w}]"] + words[i + 1:i + 7])
                        sheet_rows[f].append({"ref": p["ref"], "word_idx": i, "word": w,
                                              "context": ctx, "auto": auto[f].get(i, "")})
                for i, v in lf.items():
                    counts[(f, v, sf[i])] += 1
                labels[f] = {str(i): v for i, v in lf.items()}
                sources[f] = {str(i): s for i, s in sf.items()}
            rows.append({"ref": p["ref"], "variety": variety, "text": p[variety],
                         "words": words, "spans": [[s, e] for _, s, e in spans],
                         "labels": labels, "sources": sources})
        out = DERIVED / f"sample_{variety}.jsonl"
        write_jsonl(out, rows)
        print(f"{variety}: {len(rows)} verses -> {out}")
        for f in FEATURES:
            c = {f"{v}/{s}": n for (ff, v, s), n in sorted(counts.items()) if ff == f}
            print(f"  {f:12s} {c}")
            write_sheet(variety, f, sheet_rows[f], a.sheet_size, rng)


if __name__ == "__main__":
    main()
