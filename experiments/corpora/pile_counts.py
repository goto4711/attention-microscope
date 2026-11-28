"""How often did Pythia see each construction? Counts in the Pile (its training data).

Uses the infini-gram API (api.infini-gram.io, index v4_piletrain_llama: exact n-gram
counts over the Pile training set). Only fixed phrases are sent, no project data.

Patterns (lower case, so counts are lower bounds; each is marker + bare verb, or be + -ing):
  english progressive   is/are/was/were/am + V-ing           (was going)
  english did + verb    did + V (do-support: did go, did come) -- same form as the Creole
                        past marker, different function
  creole progressive    mi/im/dem/wi + a + V                 (mi a go)
  creole past           mi/im/dem/wi + did/en + V            (dem did go)
  creole plural         di + N + dem                          (di man dem)
  aae habitual be       he/she/they + be + V-ing              (he be working)
over a fixed list of common verbs (Creole spellings in both JNT orthography and the
English-based spelling usual online).

Pythia saw the Pile about once over 143,000 steps (2,097,152 tokens per step), in a fixed
shuffled order, so the expected number of occurrences seen by step s is about
count * s / 143000. The output gives that for the checkpoints in step 6.

  python experiments/corpora/pile_counts.py

Writes specimens/pile_counts.json (counts only); caches answers in data/derived/.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

API = "https://api.infini-gram.io/"
INDEX = "v4_piletrain_llama"
CACHE = Path("data/derived/pile_counts_cache.json")
STEPS = [0, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]
TOTAL_STEPS = 143000

VERBS = [  # (english, english -ing, creole spellings)
    ("go", "going", ["go"]), ("come", "coming", ["come", "kom"]), ("say", "saying", ["say", "se"]),
    ("tell", "telling", ["tell", "tel"]), ("do", "doing", ["do", "du"]), ("make", "making", ["make", "mek"]),
    ("take", "taking", ["take", "tek"]), ("look", "looking", ["look", "luk"]), ("work", "working", ["work", "wok"]),
    ("run", "running", ["run", "ron"]), ("talk", "talking", ["talk", "taak"]), ("want", "wanting", ["want", "waan"]),
    ("eat", "eating", ["eat", "nyam"]), ("give", "giving", ["give", "gi"]), ("see", "seeing", ["see", "si"]),
]
NOUNS = ["man", "pikni", "bwai", "gyal", "ting", "smaddi", "pipl", "uman"]


def count(q, cache):
    if q in cache:
        return cache[q]
    body = json.dumps({"index": INDEX, "query_type": "count", "query": q}).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            if "count" in d:
                cache[q] = int(d["count"])
                time.sleep(1.0)  # be polite: the API rate-limits bursts
                return cache[q]
            raise RuntimeError(d)
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(2 + 3 * attempt)
            print("retry", q, e, flush=True)


def main():
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    groups = {
        "english_progressive": [f"{b} {ing}" for b in ("is", "are", "was", "were", "am") for _, ing, _ in VERBS],
        "english_did_verb": [f"did {v}" for v, _, _ in VERBS],
        "creole_progressive": [f"{p} a {c}" for p in ("mi", "im", "dem", "wi")
                               for _, _, cs in VERBS[:10] for c in cs],
        "creole_past": [f"{p} {m} {c}" for p in ("mi", "im", "dem", "wi")
                        for m in ("did", "en") for _, _, cs in VERBS[:10] for c in cs],
        "creole_plural": [f"di {n} dem" for n in NOUNS],
        "aae_habitual_be": [f"{p} be {ing}" for p in ("he", "she", "they") for _, ing, _ in VERBS[:10]],
    }
    res = {"index": INDEX, "note": "exact n-gram counts, lower case only; lower bounds for each "
                                   "construction (fixed verb list)", "total_steps": TOTAL_STEPS,
           "groups": {}}
    for g, qs in groups.items():
        per = {}
        for q in qs:
            per[q] = count(q, cache)
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(cache))
        tot = sum(per.values())
        res["groups"][g] = {"total": tot, "n_patterns": len(qs),
                            "top": sorted(per.items(), key=lambda kv: -kv[1])[:10],
                            "seen_by_step": {s: round(tot * s / TOTAL_STEPS, 1) for s in STEPS}}
        print(f"{g:22s} {tot:>12,d}  top: {res['groups'][g]['top'][:4]}", flush=True)
    Path("specimens/pile_counts.json").write_text(json.dumps(res, indent=1))
    print("-> specimens/pile_counts.json")


if __name__ == "__main__":
    main()
