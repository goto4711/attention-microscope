"""Compare hand annotation (gold) with the automatic labels (spaCy for WEB, rules for JNT).

The annotation sheets were sampled in three strata of equal size: words the automatic
labeller marked 1, words it marked 0, and content words it left unlabelled. So the report
gives agreement per stratum rather than one pooled accuracy:

  auto 1  -> share that gold also calls 1 (precision of positives)
  auto 0  -> share that gold also calls 0
  auto -  -> share that gold calls 1 (missed positives), 0, or skip (not a candidate)

plus Cohen's kappa on the words both label 0/1. Writes specimens/annotation_report.json
(counts only, no text) and prints a table.

  python experiments/bible/annotation_report.py [--annotator NAME]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from common import FEATURES, VARIETIES, write_json

ANN = Path("data/annotation")
SRC = {"web": "spaCy en_core_web_sm", "jnt": "rules on overt markers"}


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa = Counter(a for a, _ in pairs)
    pb = Counter(b for _, b in pairs)
    pe = sum(pa[k] * pb[k] for k in (0, 1)) / n ** 2
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", default="")
    ap.add_argument("--note", default="", help="provenance note recorded with the report")
    a = ap.parse_args()
    report = {"annotator": a.annotator, "provenance": a.note, "note": "sheets sampled in equal strata by automatic label", "varieties": {}}
    for v in VARIETIES:
        rv = {}
        for f in FEATURES:
            p = ANN / f"{v}_{f}.csv"
            if not p.exists():
                continue
            rows = [r for r in csv.DictReader(open(p, encoding="utf-8")) if r["gold"].strip()]
            if not rows:
                continue
            tab = Counter((r["auto"] or "-", r["gold"]) for r in rows)
            strata = {}
            for auto in ("1", "0", "-"):
                n = sum(c for (x, _), c in tab.items() if x == auto)
                if n:
                    strata[auto] = {"n": n, **{g: round(tab[(auto, g)] / n, 3) for g in ("1", "0", "x")}}
            both = [(int(r["auto"]), int(r["gold"])) for r in rows
                    if r["auto"] in ("0", "1") and r["gold"] in ("0", "1")]
            rv[f] = {"labeller": SRC[v], "n_gold": len(rows), "strata": strata,
                     "agreement": round(sum(x == y for x, y in both) / len(both), 3) if both else None,
                     "kappa": kappa(both), "n_both": len(both),
                     "counts": {f"{x}|{y}": c for (x, y), c in sorted(tab.items())}}
        if rv:
            report["varieties"][v] = rv
    write_json("specimens/annotation_report.json", report)

    for v, rv in report["varieties"].items():
        print(f"\n{v.upper()} vs {SRC[v]}  (gold = hand labels; 1 present, 0 absent, x skip)")
        print(f"{'feature':12s} {'auto=1: gold 1':>15s} {'auto=0: gold 0':>15s} {'auto=-: gold 1/0/x':>22s} "
              f"{'agree':>6s} {'kappa':>6s}")
        for f, r in rv.items():
            s = r["strata"]
            g1 = f"{s['1']['1']:.0%} of {s['1']['n']}" if "1" in s else "-"
            g0 = f"{s['0']['0']:.0%} of {s['0']['n']}" if "0" in s else "-"
            gm = (f"{s['-']['1']:.0%}/{s['-']['0']:.0%}/{s['-']['x']:.0%} of {s['-']['n']}"
                  if "-" in s else "-")
            print(f"{f:12s} {g1:>15s} {g0:>15s} {gm:>22s} {r['agreement'] or 0:6.1%} {r['kappa']}")
    print("\n-> specimens/annotation_report.json")


if __name__ == "__main__":
    main()
