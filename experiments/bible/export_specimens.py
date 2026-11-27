"""Step 7: export real specimens for the page (index.html, "Pythia" mode).

Writes specimens/real-specimens.js, which sets window.REAL_SPECIMENS. A script file rather
than JSON so the page also works opened from disk (file://), where fetch() is blocked.

Contents:
  pairs     verse pairs (WEB and JNT): words, feature labels, word-level attention for the
            selected heads (0-100 integers), and per labelled word a readout per layer:
            the projection onto the step-3 probe direction, scaled so that +1 is the mean of
            positive words and -1 the mean of negative words at that layer
  heads     the selected heads ([layer, head, why])
  maps      per feature and variety, [L][H] grids: marker attention, head decoding,
            visibility through W_Q, W_K, W_V
  probes    per feature and variety, probe accuracy by layer and transfer AUC
  ablation  per feature and variety, the step-4 ablation runs

The JNT words are © Bible Society of the West Indies: the output is gitignored. Publish
it only once the licence allows quoting these verses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import DERIVED, FEATURES, OUT, VARIETIES, read_jsonl


def r2(a, nd=2):
    return np.round(np.asarray(a, float), nd).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-410m")
    ap.add_argument("--n", type=int, default=8, help="verse pairs to export")
    ap.add_argument("--max-words", type=int, default=28)
    ap.add_argument("--heads", type=int, default=16, help="heads whose attention is exported")
    ap.add_argument("--out", default="specimens/real-specimens.js")
    a = ap.parse_args()
    out = OUT / a.model
    meta = json.loads((out / "meta.json").read_text())
    L, H = meta["n_layers"], meta["n_heads"]
    probes_json = json.loads(Path(f"specimens/bible-pair_{a.model}_probes.json").read_text())
    heads_json = json.loads(Path(f"specimens/bible-pair_{a.model}_heads.json").read_text())
    hz = dict(np.load(out / "heads.npz"))
    P = dict(np.load(out / "probes.npz"))
    readout = heads_json["readout_layer"]

    # --- heads: top marker-attention heads, then top decoding heads ----------
    chosen, why = [], {}
    def add(l, h, reason):
        if (l, h) not in why and len(chosen) < a.heads:
            chosen.append((l, h))
            why[(l, h)] = reason
        elif (l, h) in why and reason not in why[(l, h)]:
            why[(l, h)] += "; " + reason
    for rank in range(3):
        for f in FEATURES:
            for v in VARIETIES:
                m = hz.get(f"marker_{f}_{v}")
                if m is not None:
                    idx = np.argsort(-m[:readout].ravel())[rank]
                    add(*divmod(int(idx), H), f"attends to the {f} marker ({v.upper()})")
    for rank in range(2):
        for f in FEATURES:
            for v in VARIETIES:
                d = hz.get(f"decode_{f}_{v}")
                if d is not None and np.isfinite(d).any():
                    idx = np.argsort(-np.nan_to_num(d[:readout], nan=-1).ravel())[rank]
                    add(*divmod(int(idx), H), f"output carries {f} ({v.upper()})")
    chosen.sort()

    # --- verse pairs ----------------------------------------------------------
    rows = {v: read_jsonl(DERIVED / f"sample_{v}.jsonl") for v in VARIETIES}
    attn = {v: np.load(out / f"attn_{v}.npz") for v in VARIETIES}
    acts = {v: np.load(out / f"acts_{v}.npz") for v in VARIETIES}
    n_attn = min(len([k for k in attn[v].files if k.startswith("a")]) for v in VARIETIES)

    # readout scaling per (feature, variety, layer) from all labelled words
    scale = {}
    for f in FEATURES:
        for v in VARIETIES:
            key = f"{f}_{v}"
            if key not in P:
                continue
            X, vi, wi = acts[v]["X"], acts[v]["verse"], acts[v]["word"]
            ys = np.array([rows[v][a_]["labels"][f].get(str(b_), -1) for a_, b_ in zip(vi, wi)])
            proj = np.einsum("nld,ld->nl", X.astype(np.float32), P[key])
            pos, neg = proj[ys == 1].mean(0), proj[ys == 0].mean(0)
            scale[key] = ((pos + neg) / 2, np.maximum((pos - neg) / 2, 1e-6))

    def score(i):
        n = sum(1 for v in VARIETIES for f in FEATURES for y in rows[v][i]["labels"][f].values() if y == 1)
        return n
    cand = [i for i in range(n_attn)
            if all(len(rows[v][i]["words"]) <= a.max_words for v in VARIETIES)
            and all(any(y == 1 for f in FEATURES for y in rows[v][i]["labels"][f].values())
                    for v in VARIETIES)]
    cand = sorted(cand, key=lambda i: -score(i))[:a.n]
    cand.sort()

    pairs = []
    for i in cand:
        pair = {"ref": rows["web"][i]["ref"]}
        for v in VARIETIES:
            row = rows[v][i]
            A = attn[v][f"a{i}"].astype(np.float32)
            entry = {"words": row["words"],
                     "labels": {f: row["labels"][f] for f in FEATURES},
                     "sources": {f: sorted(set(row["sources"][f].values())) for f in FEATURES},
                     "attn": [np.rint(A[l, h] * 100).astype(int).tolist() for l, h in chosen],
                     "readout": {}}
            sel = acts[v]["verse"] == i
            for k in np.where(sel)[0]:
                w = int(acts[v]["word"][k])
                rd = {}
                for f in FEATURES:
                    key = f"{f}_{v}"
                    if str(w) in row["labels"][f] and key in scale:
                        proj = np.einsum("ld,ld->l", acts[v]["X"][k].astype(np.float32), P[key])
                        mid, sc = scale[key]
                        rd[f] = r2((proj - mid) / sc)
                if rd:
                    entry["readout"][str(w)] = rd
            pair[v] = entry
        pairs.append(pair)

    maps, probes, ablation = {}, {}, {}
    for f in FEATURES:
        for v in VARIETIES:
            key = f"{f}_{v}"
            m = {}
            for name, k in (("marker", f"marker_{key}"), ("decode", f"decode_{key}"),
                            ("visQ", f"visQ_{key}"), ("visK", f"visK_{key}"), ("visV", f"visV_{key}")):
                if k in hz and np.isfinite(hz[k]).any():
                    m[name] = r2(np.nan_to_num(hz[k], nan=0), 3 if name == "marker" else 2)
            maps[key] = m
            pj = probes_json["features"][f][v]
            probes[key] = {k: pj[k] for k in pj if k in ("acc", "n_pos", "n_neg", "label_sources")
                           or k.startswith("transfer")}
            ablation[key] = heads_json["features"][f][v].get("ablation", {})

    data = {"model": meta["model"], "n_layers": L, "n_heads": H, "readout_layer": readout,
            "features": list(FEATURES), "varieties": {"web": "World English Bible",
                                                      "jnt": "Jamaican New Testament"},
            "label_note": "Labels are automatic (spaCy for WEB, rules for JNT) until hand annotation is in.",
            "credit": "Jamaican New Testament © 2012 Bible Society of the West Indies; "
                      "World English Bible, public domain (ebible.org).",
            "heads": [[l, h, why[(l, h)]] for l, h in chosen],
            "pairs": pairs, "maps": maps, "probes": probes, "ablation": ablation,
            "cosine": {f: probes_json["features"][f].get("cosine_web_jnt") for f in FEATURES}}
    js = "window.REAL_SPECIMENS=" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n"
    Path(a.out).write_text(js, encoding="utf-8")
    print(f"{len(pairs)} pairs, {len(chosen)} heads -> {a.out} ({len(js) / 1024:.0f} KB)")
    print("heads:", [(l, h) for l, h in chosen])


if __name__ == "__main__":
    main()
