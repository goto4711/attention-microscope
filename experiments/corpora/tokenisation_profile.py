"""Step 5 control: how Pythia's tokenizer cuts up each collection.

For every collection in data/ (and spoken vs written where the collection has both):
tokens per word, the share of words that are a single token, and the share of words
split into three or more pieces. A word is a run of letters (with internal ' or -); each
word is tokenised as " word", as it appears inside running text.

Light cleaning per format:
  ICE (Jamaica, India)  editorial <&>…</&>, untranscribed <O>…</O> and extra-corpus
                        <X>…</X> removed, then all markup; S* files spoken, W* written
  CORAAL                Content column of the interviewees only (speaker ids with _int_
                        are interviewers); /RD-…/ redactions and <…> / […] / (…) removed
  AAPB oral histories   [bracketed] notes, ALL-CAPS interviewer turns and speaker labels removed
  WPA narratives        Project Gutenberg header and licence removed
  LinCE Hindi–English   word lists from the Arrow files; also by language tag
  WEB, JNT              the Bible pair (derived verses)

Writes specimens/tokenisation_profile.json (numbers only) and prints a table, with the
distance from Standard English for each collection as given in the project plan.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.abspath("data/hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*")
D = Path("data")

TAGS = {"lang1": "English", "lang2": "Hindi (romanised)", "ne": "named entity", "other": "other",
        "mixed": "mixed", "ambiguous": "ambiguous", "fw": "foreign word", "unk": "unknown",
        "?": "untagged (test split)", "None": "untagged (test split)"}

# distance from Standard English, as set in the project plan
DISTANCE = {"web": 0.0, "ice-ja-written": 0.20, "ice-ja-spoken": 0.35, "aapb-howard": 0.26,
            "aapb-eotp": 0.26, "ice-in-written": 0.34, "ice-in-spoken": 0.34, "wpa": 0.37,
            "coraal": 0.37, "lince": 0.57, "jnt": 0.89}
LABEL = {"web": "World English Bible", "ice-ja-written": "ICE Jamaica, written",
         "ice-ja-spoken": "ICE Jamaica, spoken", "aapb-howard": "Howard Univ. TV oral histories",
         "aapb-eotp": "AAPB interviews (eotp-interviews)", "ice-in-written": "ICE India, written",
         "ice-in-spoken": "ICE India, spoken", "wpa": "WPA narratives (1930s)",
         "coraal": "CORAAL (DCA, ROC)", "lince": "LinCE Hindi–English", "jnt": "Jamaican New Testament"}


def ice(folder, mode):
    for f in sorted(glob.glob(str(D / folder / "*"))):
        name = Path(f).name.upper()
        if not name.endswith(".TXT") or name[0] != mode:
            continue
        t = open(f, encoding="latin-1").read()
        t = re.sub(r"<&>.*?</&>|<O>.*?</O>|<X>.*?</X>", " ", t, flags=re.S | re.I)
        yield re.sub(r"<[^>]*>", " ", t)


def coraal():
    for f in sorted(glob.glob(str(D / "coraal/*/*.txt"))):
        for line in open(f, encoding="utf-8", errors="replace").read().splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 4 and "_int_" not in parts[1]:
                t = re.sub(r"/[^/]*/|<[^>]*>|\[[^\]]*\]|\([^)]*\)", " ", parts[3])
                yield t.replace("[", " ").replace("]", " ")


def aapb(folder):
    for f in sorted(glob.glob(str(D / folder / "*.txt"))):
        t = open(f, encoding="utf-8", errors="replace").read()
        t = re.sub(r"\[[^\]]*\]", " ", t)
        t = re.sub(r"\b\d{2}:\d{2}:\d{2}(?::\d{2})?\b", " ", t)          # timecodes
        t = re.sub(r"\b(?:INTERVIEWER|INTERVIEWEE|CAMERA[A-Z ]*)\s*\d*\s*:", " ", t)
        t = re.sub(r"(?<![\w])\d+\s*:", " ", t)
        # drop interviewer turns transcribed in capitals (runs of 3+ all-caps words)
        t = re.sub(r"(?:\b[A-Z][A-Z'’-]+\b[\s,.?!:;—-]*){3,}", " ", t)
        # speaker labels such as "Clark:" or "JUST A McLaurin:"
        t = re.sub(r"(?:\b[A-Z][\w.'’-]*\s){0,3}[A-Z][\w.'’-]*:\s", " ", t)
        yield t


def wpa():
    for f in sorted(glob.glob(str(D / "wpa_narratives/*.txt"))):
        t = open(f, encoding="utf-8", errors="replace").read()
        s = re.search(r"\*\*\* ?START OF.*?\*\*\*", t)
        e = re.search(r"\*\*\* ?END OF", t)
        yield t[s.end() if s else 0:e.start() if e else len(t)]


def lince():
    import pyarrow as pa
    for f in sorted(glob.glob(str(D / "lince_hf/*lid_hineng*/**/*.arrow"), recursive=True)):
        with pa.memory_map(f) as src:
            try:
                tab = pa.ipc.open_stream(src).read_all()
            except Exception:
                tab = pa.ipc.open_file(src).read_all()
        cols = tab.column_names
        wcol = next(c for c in cols if c.lower() in ("words", "tokens", "sentence"))
        lcol = next((c for c in cols if c.lower() in ("lid", "labels", "lang", "tags")), None)
        words = tab.column(wcol).to_pylist()
        labs = tab.column(lcol).to_pylist() if lcol else [None] * len(words)
        for ws, ls in zip(words, labs):
            yield ws, ls


def bible(v):
    if v == "web":
        nt = re.compile(r"^(MAT|MAR|LUK|JOH|ACT|ROM|1CO|2CO|GAL|EPH|PHI|COL|1TH|2TH|1TI|2TI|TIT|PHM|HEB|JAM|1PE|2PE|1JO|2JO|3JO|JUD|REV) ")
        for line in open(D / "web/eng-web_vpl.txt", encoding="utf-8"):
            if nt.match(line):
                yield line.split(" ", 2)[2]
    else:
        for line in open(D / "derived/jnt_verses.jsonl", encoding="utf-8"):
            yield json.loads(line)["text"]


def profile(words, tok, cache):
    n = Counter()
    for w in words:
        if w not in cache:
            cache[w] = len(tok(" " + w, add_special_tokens=False)["input_ids"])
        n[cache[w]] += 1
    tot = sum(n.values())
    if not tot:
        return None
    return {"words": tot, "types": None,
            "tokens_per_word": round(sum(k * c for k, c in n.items()) / tot, 3),
            "single_token": round(n[1] / tot, 3),
            "three_plus": round(sum(c for k, c in n.items() if k >= 3) / tot, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-words", type=int, default=300000, help="cap per collection")
    ap.add_argument("--tokenizer", default="EleutherAI/pythia-410m")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    cache = {}

    def words_of(texts):
        out = []
        for t in texts:
            out.extend(WORD.findall(t))
            if len(out) >= a.max_words:
                break
        return out[:a.max_words]

    sources = {
        "web": lambda: words_of(bible("web")),
        "ice-ja-written": lambda: words_of(ice("ice-jamaica/CORPUS", "W")),
        "ice-ja-spoken": lambda: words_of(ice("ice-jamaica/CORPUS", "S")),
        "aapb-howard": lambda: words_of(aapb("storycorps")),
        "aapb-eotp": lambda: words_of(aapb("aapb/eotp-interviews")),
        "ice-in-written": lambda: words_of(ice("ice-india/Corpus", "W")),
        "ice-in-spoken": lambda: words_of(ice("ice-india/Corpus", "S")),
        "wpa": lambda: words_of(wpa()),
        "coraal": lambda: words_of(coraal()),
        "jnt": lambda: words_of(bible("jnt")),
    }
    res = {"tokenizer": a.tokenizer, "word": "run of letters with internal ' or -, tokenised as ' word'",
           "max_words": a.max_words, "collections": {}}
    for key, fn in sources.items():
        try:
            ws = fn()
        except Exception as e:  # a missing folder should not stop the rest
            print(f"{key}: skipped ({e})")
            continue
        p = profile(ws, tok, cache)
        if p:
            p["types"] = len(set(ws))
            res["collections"][key] = {"label": LABEL[key], "distance": DISTANCE[key], **p}

    try:
        allw, by = [], Counter()
        bylang = {}
        for ws, ls in lince():
            for i, w in enumerate(ws):
                if WORD.fullmatch(w or ""):
                    allw.append(w)
                    lab = TAGS.get(str(ls[i]) if ls else "?", str(ls[i]) if ls else "?")
                    bylang.setdefault(lab, []).append(w)
            if len(allw) >= a.max_words:
                break
        p = profile(allw, tok, cache)
        p["types"] = len(set(allw))
        p["by_language_tag"] = {lab: profile(ws, tok, cache) for lab, ws in bylang.items() if len(ws) >= 500}
        res["collections"]["lince"] = {"label": LABEL["lince"], "distance": DISTANCE["lince"], **p}
    except Exception as e:
        print(f"lince: skipped ({e})")

    Path("specimens").mkdir(exist_ok=True)
    json.dump(res, open("specimens/tokenisation_profile.json", "w"), indent=1, ensure_ascii=False)
    print(f"{'collection':36s} {'dist':>5s} {'words':>8s} {'tok/word':>8s} {'1 tok':>6s} {'3+ tok':>6s}")
    for k, c in sorted(res["collections"].items(), key=lambda kv: kv[1]["distance"]):
        print(f"{c['label']:36s} {c['distance']:5.2f} {c['words']:8d} {c['tokens_per_word']:8.2f} "
              f"{c['single_token']:6.0%} {c['three_plus']:6.0%}")
        for lab, q in (c.get("by_language_tag") or {}).items():
            if q:
                print(f"   tag {lab:30s} {'':5s} {q['words']:8d} {q['tokens_per_word']:8.2f} "
                      f"{q['single_token']:6.0%} {q['three_plus']:6.0%}")
    print("-> specimens/tokenisation_profile.json")


if __name__ == "__main__":
    main()
