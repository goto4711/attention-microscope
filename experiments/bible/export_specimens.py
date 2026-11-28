"""Step 7: export real specimens for the page (index.html, "Pythia" mode).

Writes specimens/real-specimens.js, which sets window.REAL_SPECIMENS. A script file rather
than JSON so the page also works opened from disk (file://), where fetch() is blocked.

Contents:
  pairs     verse pairs (WEB and JNT): words, feature labels, word-level attention for the
            selected heads (0-100 integers), and per labelled word a readout per layer:
            the projection onto the step-3 probe direction, scaled so that +1 is the mean of
            positive words and -1 the mean of negative words at that layer
  heads     the selected heads ([layer, head, why])
  maps      per feature and variety, [L][H] grids: marker attention, marker attention beyond
            what the head gives any word at that distance ("excess"), head decoding,
            visibility through W_Q, W_K, W_V
  prevword  per variety, [L][H] attention to the word directly before (any word)
  controls  ablation of generic previous-word heads (step 5) and the respelled JNT
            (orthography control): probe accuracy and marker-head ablation
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


def label_note(rows):
    n = {v: sum(1 for r in rows[v] for f in FEATURES for s in r["sources"][f].values() if s == "gold")
         for v in VARIETIES}
    return (f"Labels: spaCy for the WEB, rules on overt markers for the JNT, replaced by hand labels "
            f"where a word was annotated ({n['web']} WEB and {n['jnt']} JNT labels).")


def collections(model):
    """Cross-collection summaries (numbers only) from compare.py, heads_collections.py and
    lince_switch.py, if they have been run."""
    def load(p):
        p = Path(p)
        return json.loads(p.read_text()) if p.exists() else {}
    cmp_ = load(f"specimens/corpora_{model}.json")
    hc = load(f"specimens/corpora_heads_{model}.json").get("collections", {})
    ls = load(f"specimens/lince_switch_{model}.json")
    out = {"mid_layers": cmp_.get("mid_layers"), "spearman": cmp_.get("summary", {}), "rows": []}
    for key, c in cmp_.get("collections", {}).items():
        row = {"key": key, "label": c["label"], "distance": c["distance"],
               "tokens_per_word": c["tokens_per_word"], "transfer": {}, "unmarked": {}}
        for f, fe in c["features"].items():
            if "transfer_mid" in fe:
                row["transfer"][f] = fe["transfer_mid"]
                row.setdefault("transfer_ci", {})[f] = fe.get("transfer_mid_ci")
            if "unmarked" in fe:
                u = fe["unmarked"]
                row["unmarked"][f] = [u["transfer_mid_marked"], u["transfer_mid_unmarked"], u["n_pos_unmarked"],
                                      u.get("transfer_mid_marked_ci"), u.get("transfer_mid_unmarked_ci")]
        h = hc.get(key, {})
        if h:
            row["be_ing_4_5"] = h.get("be_ing", {}).get("head_4_5")
            row["prev_ablation_k20"] = {f: fe["runs"][-1]["prev_word_heads"]
                                        for f, fe in h.get("ablation", {}).items() if "runs" in fe}
        out["rows"].append(row)
    if ls:
        best = lambda k: max(ls[k]["acc"])
        out["lince"] = {"sentences": ls["sentences"], "tokenisation": ls["tokenisation"],
                        **{k: {"best": best(k), "baselines": ls[k]["baselines"]}
                           for k in ("language", "switch", "anticipation")}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-410m")
    ap.add_argument("--n", type=int, default=8, help="verse pairs to export")
    ap.add_argument("--max-words", type=int, default=28)
    ap.add_argument("--heads", type=int, default=16, help="heads whose attention is exported")
    ap.add_argument("--out", default=None)
    ap.add_argument("--public", action="store_true",
                    help="no JNT text: pairs keep only the WEB verse; writes specimens/public-specimens.js")
    a = ap.parse_args()
    a.out = a.out or ("specimens/public-specimens.js" if a.public else "specimens/real-specimens.js")
    out = OUT / a.model
    meta = json.loads((out / "meta.json").read_text())
    L, H = meta["n_layers"], meta["n_heads"]
    probes_json = json.loads(Path(f"specimens/bible-pair_{a.model}_probes.json").read_text())
    heads_json = json.loads(Path(f"specimens/bible-pair_{a.model}_heads.json").read_text())
    hz = dict(np.load(out / "heads.npz"))
    P = dict(np.load(out / "probes.npz"))
    readout = heads_json["readout_layer"]

    rows = {v: read_jsonl(DERIVED / f"sample_{v}.jsonl") for v in VARIETIES}
    attn = {v: np.load(out / f"attn_{v}.npz") for v in VARIETIES}
    acts = {v: np.load(out / f"acts_{v}.npz") for v in VARIETIES}
    n_attn = min(len([k for k in attn[v].files if k.startswith("a")]) for v in VARIETIES)
    ctrl_path = Path(f"specimens/bible-pair_{a.model}_controls.json")
    orth_path = Path(f"specimens/bible-pair_{a.model}_orthography.json")
    controls = json.loads(ctrl_path.read_text()) if ctrl_path.exists() else {}
    orth = json.loads(orth_path.read_text()) if orth_path.exists() else {}

    # --- generic attention by distance (1-4 words back), from the saved word-level attention
    gen = {}
    for v in VARIETIES:
        acc, cnt = np.zeros((5, L, H)), np.zeros(5)
        for i in range(n_attn):
            A = attn[v][f"a{i}"].astype(np.float32)  # [L, H, W, W]
            for q in range(A.shape[2]):
                for k in range(1, 5):
                    if q - k >= 0:
                        acc[k] += A[:, :, q, q - k]
                        cnt[k] += 1
        gen[v] = acc / np.maximum(cnt, 1)[:, None, None]
    prev = (gen["web"][1] + gen["jnt"][1]) / 2
    prev_rank = np.zeros((L, H), int)
    prev_rank.ravel()[np.argsort(-prev.ravel())] = np.arange(1, L * H + 1)

    # marker attention beyond what the head gives any word at the same distance
    excess = {}
    for f in FEATURES:
        for v in VARIETIES:
            m = hz.get(f"marker_{f}_{v}")
            off = controls.get("dynamic", {}).get("offset", {}).get(f"{f}_{v}", {}).get("offsets")
            if m is None or not off:
                continue
            n = sum(off.values())
            expect = sum(c * gen[v][int(k)] for k, c in off.items() if int(k) < 5) / n
            excess[f"{f}_{v}"] = (m - expect, expect)

    # --- heads: marker attention, attention beyond the previous-word habit, decoding
    chosen, why = [], {}
    def add(l, h, reason):
        if (l, h) not in why and len(chosen) < a.heads:
            chosen.append((l, h))
            why[(l, h)] = [reason]
        elif (l, h) in why and reason not in why[(l, h)]:
            why[(l, h)].append(reason)
    for rank in range(3):
        for f in FEATURES:
            for v in VARIETIES:
                m = hz.get(f"marker_{f}_{v}")
                if m is not None:
                    l, h = divmod(int(np.argsort(-m[:readout].ravel())[rank]), H)
                    add(l, h, f"{round(m[l, h] * 100)}% to the {f} marker in {v.upper()}")
    for f in FEATURES:
        for v in VARIETIES:
            if f"{f}_{v}" in excess:
                ex, _ = excess[f"{f}_{v}"]
                l, h = divmod(int(np.argsort(-ex[:readout].ravel())[0]), H)
                add(l, h, f"{round(hz[f'marker_{f}_{v}'][l, h] * 100)}% to the {f} marker in {v.upper()}")
    for rank in range(2):
        l, h = divmod(int(np.argsort(-prev[:readout].ravel())[rank]), H)
        add(l, h, "previous-word head")
    for f in FEATURES:
        for v in VARIETIES:
            d = hz.get(f"decode_{f}_{v}")
            if d is not None and np.isfinite(d).any():
                l, h = divmod(int(np.argsort(-np.nan_to_num(d[:readout], nan=-1).ravel())[0]), H)
                add(l, h, f"output carries {f} ({v.upper()})")
    chosen.sort()
    def describe(l, h):
        parts = why[(l, h)]
        g = round(prev[l, h] * 100)
        lead = (f"generic previous-word head (rank {prev_rank[l, h]} of {L * H})"
                if prev_rank[l, h] <= 12 else "not a previous-word head")
        return lead + "; " + "; ".join(p for p in parts if p != "previous-word head")

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
            if key in excess:
                m["excess"] = r2(excess[key][0], 3)
            maps[key] = m
            pj = probes_json["features"][f][v]
            probes[key] = {k: pj[k] for k in pj if k in ("acc", "n_pos", "n_neg", "label_sources")
                           or k.startswith("transfer")}
            ablation[key] = heads_json["features"][f][v].get("ablation", {})

    data = {"model": meta["model"], "n_layers": L, "n_heads": H, "readout_layer": readout,
            "features": list(FEATURES), "varieties": {"web": "World English Bible",
                                                      "jnt": "Jamaican New Testament"},
            "label_note": label_note(rows),
            "credit": "Jamaican New Testament © 2012 Bible Society of the West Indies; "
                      "World English Bible, public domain (ebible.org).",
            "heads": [[l, h, describe(l, h).rstrip("; ")] for l, h in chosen],
            "prevword": {v: r2(gen[v][1], 3) for v in VARIETIES}, "prev_rank": prev_rank.tolist(),
            "prev_ablation": controls.get("dynamic", {}).get("prev_ablation", {}),
            "orthography": {"tokenisation": orth.get("tokenisation", {}),
                            "features": {f: {k: v for k, v in e.items()
                                             if k in ("acc_jntr", "ablation")}
                                         for f, e in orth.get("features", {}).items()}},
            "pairs": pairs, "maps": maps, "probes": probes, "ablation": ablation,
            "cosine": {f: probes_json["features"][f].get("cosine_web_jnt") for f in FEATURES},
            "selectivity": {k: c for k, c in controls.get("static", {}).get("control_task", {}).items()},
            "collections": collections(a.model)}
    name = "REAL_SPECIMENS"
    if a.public:  # the WEB is public domain; the JNT verses are not
        data["pairs"] = [{"ref": p["ref"], "web": p["web"]} for p in pairs]
        data["public"] = True
        data["credit"] = ("World English Bible, public domain (ebible.org). The Jamaican New Testament "
                          "(© 2012 Bible Society of the West Indies) is analysed but not quoted here.")
        name = "PUBLIC_SPECIMENS"
    js = f"window.{name}=" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n"
    Path(a.out).write_text(js, encoding="utf-8")
    print(f"{len(pairs)} pairs, {len(chosen)} heads -> {a.out} ({len(js) / 1024:.0f} KB)")
    print("heads:", [(l, h) for l, h in chosen])


if __name__ == "__main__":
    main()
