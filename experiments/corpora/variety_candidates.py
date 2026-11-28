"""Candidates for variety-specific constructions that a Standard-English tagger cannot label.

spaCy labels plural, past and progressive only where they carry the standard marking
(-s, -ed, be + -ing). The constructions below mean the same but are marked differently
or not at all. This script finds candidates for them across the whole collection, adds
the candidate sentences to data/derived/sample_<collection>.jsonl (so extract.py covers
them), and writes data/annotation/variety_<collection>.csv for hand annotation.

  kind               candidate                                              feature if yes
  plural_unmarked    singular-tagged noun after a number > 1, a plural     plural = 1
                     quantifier or determiner (two mile, them boy), or
                     before Creole plural dem (di bwai dem)
  past_unmarked      present/base-tagged verb in a past context (a past    past = 1
                     verb or past time word in the sentence), or after
                     Creole did / en / ben
  progressive_nobe   -ing verb without a form of be (she going), or verb   progressive = 1
                     after Creole a / de (im a go)
  habitual_be        invariant be after a subject, not after a modal, to,  habitual = 1
                     or do (they be working)
  copula_absence     subject pronoun directly before a predicate (-ing,     copula_absence = 1
                     adjective, gonna, place) with no be (he tall)
  agreement_3sg      third-person singular subject with an unmarked        agreement_unmarked = 1
                     present-tense verb (he go, she have)

Rows of the same kind are capped per collection (--per-kind), sampled with a seed.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bible"))
sys.path.insert(0, str(HERE))
from common import DERIVED, FEATURES, read_jsonl, words_with_spans, write_jsonl  # noqa: E402
from features import web_labels  # noqa: E402
from build_samples import COLLECTIONS, sentences  # noqa: E402

KINDS = ["plural_unmarked", "past_unmarked", "progressive_nobe", "habitual_be",
         "copula_absence", "agreement_3sg"]
NUM_GT1 = {"two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
           "twelve", "twenty", "thirty", "forty", "fifty", "hundred", "thousand", "million",
           "several", "many", "few", "both", "these", "those", "them", "dem",
           "couple", "dozen", "various", "numerous"}
PAST_WORDS = {"yesterday", "ago", "last", "was", "were", "had", "did", "used", "back", "then",
              "when", "once"}
JA_PAST = {"did", "en", "ben", "neehn", "neva"}
JA_PROG = {"a", "de"}
BE = {"be", "is", "are", "was", "were", "am", "been", "being", "'s", "'re", "'m", "s", "re", "m"}
SUBJ3 = {"he", "she", "it"}
SUBJ = {"i", "you", "he", "she", "it", "we", "they", "dey", "y'all"}
PRED_WORDS = {"gonna", "finna", "here", "there", "in", "at", "from", "real", "so", "too", "not",
              "gone", "done", "sick", "tired", "good", "bad", "happy", "mad", "home"}


def find(doc, creole):
    """Yield (token index in doc, kind) for candidate constructions."""
    toks = list(doc)
    low = [t.lower_ for t in toks]
    past_ctx = any(t.tag_ == "VBD" for t in toks) or any(w in PAST_WORDS for w in low)
    for i, t in enumerate(toks):
        prev = low[i - 1] if i else ""
        prev2 = low[i - 2] if i > 1 else ""
        nxt = low[i + 1] if i + 1 < len(toks) else ""
        if t.tag_ == "NN" and (prev in NUM_GT1 or (prev2 in NUM_GT1 and toks[i - 1].pos_ == "ADJ")
                               or (creole and nxt == "dem")):
            yield i, "plural_unmarked"
        if t.pos_ in ("VERB", "AUX") and t.tag_ in ("VB", "VBP", "VBZ") and t.lower_ not in BE:
            after_modal = prev in {"to", "will", "would", "can", "could", "should", "must", "might",
                                   "may", "shall", "do", "does", "don't", "did", "gonna", "let"}
            if creole and prev in JA_PAST:
                yield i, "past_unmarked"
            elif past_ctx and not after_modal and t.tag_ in ("VB", "VBP") \
                    and not (prev == "you" and t.lower_ in ("know", "see", "mean")) \
                    and prev2 not in {"didn't", "don't", "doesn't", "never"}:
                yield i, "past_unmarked"
        if t.tag_ == "VBG" and t.pos_ == "VERB" and prev not in BE and prev2 not in BE \
                and (prev in SUBJ or toks[i - 1].pos_ in ("PRON", "PROPN", "NOUN") if i else False):
            yield i, "progressive_nobe"
        if creole and prev in JA_PROG and t.pos_ == "VERB" and prev2 in SUBJ | {"dem", "im", "mi", "unu"}:
            yield i, "progressive_nobe"
        if t.lower_ == "be" and i > 0 and prev not in {"to", "will", "would", "can", "could",
                                                       "should", "must", "might", "may", "don't",
                                                       "do", "not", "'ll", "ll", "gonna", "let"} \
                and toks[i - 1].pos_ in ("PRON", "NOUN", "PROPN", "ADV"):
            yield i, "habitual_be"
        if prev in SUBJ and i > 0 and t.pos_ != "AUX" and t.lower_ not in BE and \
                t.tag_ != "VBG" and (t.tag_ == "JJ" or low[i] in PRED_WORDS):
            yield i, "copula_absence"
        if prev in SUBJ3 and t.tag_ in ("VBP", "VB") and t.pos_ == "VERB" and t.lower_ not in BE \
                and not t.text.endswith("-") and prev2 not in {"did", "does", "do", "didn't", "doesn't",
                                                              "can", "will", "would", "could", "should",
                                                              "let", "make", "made", "help", "saw", "see"}:
            yield i, "agreement_3sg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-kind", type=int, default=80)
    ap.add_argument("--max-words", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    import spacy
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2_000_000
    keys = a.only or [k for k in COLLECTIONS if k != "lince"]
    for key in keys:
        rng = random.Random(a.seed)
        creole = key.startswith("ice-ja")
        sp = DERIVED / f"sample_{key}.jsonl"
        rows = [r for r in read_jsonl(sp) if not r["ref"].split(":")[1].startswith("x")]
        have = {r["text"] for r in rows}
        pool = sentences(nlp, list(COLLECTIONS[key]()), random.Random(a.seed), max_words=a.max_words)
        found = {k: [] for k in KINDS}
        for doc, txt in zip(nlp.pipe(pool, batch_size=64), pool):
            spans = words_with_spans(txt)
            start2w = {s: wi for wi, (_, s, _) in enumerate(spans)}
            for ti, kind in find(doc, creole):
                wi = start2w.get(doc[ti].idx)
                if wi is not None:
                    found[kind].append((txt, wi))
        picked = []
        for k in KINDS:
            c = found[k]
            rng.shuffle(c)
            seen = set()
            for txt, wi in c:
                if (txt, wi) not in seen and len([p for p in picked if p[2] == k]) < a.per_kind:
                    seen.add((txt, wi))
                    picked.append((txt, wi, k))
        # add candidate sentences to the sample (refs "<key>:x<n>")
        extra, ref_of = [], {r["text"]: r["ref"] for r in rows}
        for txt, _, _ in picked:
            if txt in ref_of:
                continue
            spans = words_with_spans(txt)
            lab = web_labels(nlp, txt, spans)
            ref = f"{key}:x{len(extra)}"
            ref_of[txt] = ref
            extra.append({"ref": ref, "variety": key, "text": txt, "words": [w for w, _, _ in spans],
                          "spans": [[s, e] for _, s, e in spans],
                          "labels": {f: {str(i): y for i, y in lab[f].items()} for f in FEATURES},
                          "sources": {f: {str(i): "spacy" for i in lab[f]} for f in FEATURES}})
        write_jsonl(sp, rows + extra)
        out = Path("data/annotation") / f"variety_{key}.csv"
        with open(out, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["ref", "word_idx", "word", "kind", "context", "sentence", "gold", "note"])
            w.writeheader()
            for txt, wi, k in sorted(picked, key=lambda p: (KINDS.index(p[2]), ref_of[p[0]])):
                words = [x for x, _, _ in words_with_spans(txt)]
                ctx = " ".join(words[max(0, wi - 8):wi] + [f"[{words[wi]}]"] + words[wi + 1:wi + 9])
                w.writerow({"ref": ref_of[txt], "word_idx": wi, "word": words[wi], "kind": k,
                            "context": ctx, "sentence": txt, "gold": "", "note": ""})
        print(f"{key:16s} pool {len(pool):6d} sentences, +{len(extra)} added to sample | " +
              ", ".join(f"{k} {min(len(found[k]), a.per_kind)}/{len(found[k])}" for k in KINDS))


if __name__ == "__main__":
    main()
