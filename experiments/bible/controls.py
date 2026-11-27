"""Step 5: controls for the Bible pair.

Static controls (from saved activations, no model run):

  tokenisation   tokens per word per variety, overall and for labelled words by class;
                 share of single-token words
  word position  probe accuracy with the word read at its first, mean or last subtoken
                 (needs extract.py --word-pos first / mean)
  single-token   probe accuracy on labelled words that are one token, last-subtoken reading
  control task   Hewitt & Liang (2019): each word type gets a random label with the same
                 base rate; selectivity = real accuracy - control accuracy. A probe that
                 only memorises word identity scores as well on the control task
  type baseline  accuracy of predicting the label from the word's own type alone, and from
                 (previous word, word): how much of the feature the surface string already
                 gives away. A reference line, not a property of the model

Dynamic controls (model run, --dynamic):

  offset         marker attention (step 4) compared with the same head's attention to the
                 word at the same distance in general: is a "marker head" just a
                 previous-word head?
  prev ablation  ablate the K heads with the highest generic previous-word attention and
                 compare the effect with step 4's marker heads

Writes specimens/bible-pair_<model>_controls.json (numbers only).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from collections import Counter, defaultdict

os.environ.setdefault("HF_HOME", os.path.abspath("data/hf_cache"))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from common import DERIVED, FEATURES, OUT, VARIETIES, read_jsonl, write_json

warnings.filterwarnings("ignore", category=UserWarning)
LAYERS = [0, 2, 4, 6, 8, 10, 12, 16, 20, 24]


def cv_acc(X, y, groups, C=0.05, folds=5):
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(folds).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    return round(float(balanced_accuracy_score(y, pred)), 3)


def cv_lookup(keys, y, groups, folds=5):
    """Predict y from a discrete key by majority in the training folds (unseen key -> majority class)."""
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(folds).split(keys, y, groups):
        by = defaultdict(Counter)
        for k, t in zip(keys[tr], y[tr]):
            by[k][t] += 1
        base = Counter(y[tr]).most_common(1)[0][0]
        # balanced: compare class frequencies relative to their base rates
        rate = {c: max(1, (y[tr] == c).sum()) for c in (0, 1)}
        for i in te:
            c = by.get(keys[i])
            pred[i] = base if not c else max((0, 1), key=lambda k: c[k] / rate[k])
    return round(float(balanced_accuracy_score(y, pred)), 3)


def load_labelled(out, v, rows, suffix=""):
    z = np.load(out / f"acts_{v}{suffix}.npz")
    return z["X"], z["verse"], z["word"]


def static(a, out, meta):
    res = {"tokenisation": {}, "word_position": {}, "single_token": {}, "control_task": {},
           "type_baseline": {}}
    rng = np.random.default_rng(0)
    for v in VARIETIES:
        rows = read_jsonl(DERIVED / f"sample_{v}.jsonl")
        z = np.load(out / f"acts_{v}.npz")
        tpw_flat = z["tokens_per_word"]
        offs = np.cumsum([0] + [len(r["words"]) for r in rows])
        tpw = lambda vi, wi: int(tpw_flat[offs[vi] + wi])
        res["tokenisation"][v] = {"tokens_per_word": round(float(tpw_flat.mean()), 3),
                                  "single_token_share": round(float((tpw_flat == 1).mean()), 3)}
        X, verse, word = z["X"], z["verse"], z["word"]
        alt = {}
        for pos in ("first", "mean"):
            p = out / f"acts_{v}_{pos}.npz"
            if p.exists():
                za = np.load(p)
                assert (za["verse"] == verse).all() and (za["word"] == word).all()
                alt[pos] = za["X"]
        for f in FEATURES:
            lab = [(k, rows[vi]["labels"][f].get(str(wi))) for k, (vi, wi) in enumerate(zip(verse, word))]
            lab = [(k, y) for k, y in lab if y is not None]
            idx = np.array([k for k, _ in lab])
            y = np.array([y for _, y in lab])
            key = f"{f}_{v}"
            if min(y.sum(), (1 - y).sum()) < 20:
                for d in res.values():
                    d[key] = {"skipped": "too few labels"}
                continue
            g = verse[idx]
            types = np.array([rows[verse[k]]["words"][word[k]].lower() for k in idx])
            prev = np.array([rows[verse[k]]["words"][word[k] - 1].lower() if word[k] else "<s>" for k in idx])
            ntok = np.array([tpw(verse[k], word[k]) for k in idx])
            res["tokenisation"][key] = {
                "pos_tokens_per_word": round(float(ntok[y == 1].mean()), 3),
                "neg_tokens_per_word": round(float(ntok[y == 0].mean()), 3),
                "pos_single_token": round(float((ntok[y == 1] == 1).mean()), 3),
                "neg_single_token": round(float((ntok[y == 0] == 1).mean()), 3)}
            # word position
            wp = {"layers": LAYERS, "last": [cv_acc(X[idx, l].astype(np.float32), y, g) for l in LAYERS]}
            for pos, XA in alt.items():
                wp[pos] = [cv_acc(XA[idx, l].astype(np.float32), y, g) for l in LAYERS]
            res["word_position"][key] = wp
            # single-token subset
            st = ntok == 1
            if min(y[st].sum(), (1 - y[st]).sum()) >= 20:
                res["single_token"][key] = {"n": int(st.sum()), "layers": LAYERS,
                                            "acc": [cv_acc(X[idx[st], l].astype(np.float32), y[st], g[st])
                                                    for l in LAYERS]}
            else:
                res["single_token"][key] = {"skipped": "too few single-token words in a class",
                                            "n": int(st.sum())}
            # control task: random label per word type, same base rate
            utypes = sorted(set(types))
            rate = y.mean()
            ctrl_map = {t: int(rng.random() < rate) for t in utypes}
            yc = np.array([ctrl_map[t] for t in types])
            if 0 < yc.sum() < len(yc):
                real = [cv_acc(X[idx, l].astype(np.float32), y, g) for l in LAYERS]
                ctrl = [cv_acc(X[idx, l].astype(np.float32), yc, g) for l in LAYERS]
                res["control_task"][key] = {"layers": LAYERS, "real": real, "control": ctrl,
                                            "selectivity": [round(r - c, 3) for r, c in zip(real, ctrl)],
                                            "n_types": len(utypes)}
            # surface baselines
            res["type_baseline"][key] = {
                "word_type": cv_lookup(types, y, g),
                "prev_and_word": cv_lookup(np.array([p + "|" + t for p, t in zip(prev, types)]), y, g),
                "prev_word": cv_lookup(prev, y, g)}
            print(f"{key:18s} tok/word pos {res['tokenisation'][key]['pos_tokens_per_word']:.2f} "
                  f"neg {res['tokenisation'][key]['neg_tokens_per_word']:.2f} | "
                  f"L12 last {wp['last'][6]:.3f} first {wp.get('first', [None]*10)[6]} "
                  f"mean {wp.get('mean', [None]*10)[6]} | single-token L12 "
                  f"{res['single_token'][key].get('acc', [None]*10)[6]} | selectivity L12 "
                  f"{res['control_task'].get(key, {}).get('selectivity', [None]*10)[6]} | "
                  f"type {res['type_baseline'][key]['word_type']} prev {res['type_baseline'][key]['prev_word']}")
    return res


def dynamic(a, out, meta):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from extract import pick_device
    from heads import Capture, marker_index, run_verse

    device = pick_device(a.device)
    L, H, hd = meta["n_layers"], meta["n_heads"], meta["d_head"]
    hz = dict(np.load(out / "heads.npz"))
    P = dict(np.load(out / "probes.npz"))
    tok = AutoTokenizer.from_pretrained(meta["model"], revision=meta["revision"])
    model = AutoModelForCausalLM.from_pretrained(meta["model"], revision=meta["revision"],
                                                 attn_implementation="eager", dtype=torch.float32)
    model.eval().to(device)
    bos = tok.eos_token_id
    cap = Capture(model, H, hd)
    r = a.readout
    res = {"readout_layer": r, "offset": {}, "prev_ablation": {}}

    # generic attention by relative offset (1..4 words back), per head, per variety
    gen = {}
    for v in VARIETIES:
        rows = read_jsonl(DERIVED / f"sample_{v}.jsonl")[:a.n]
        acc = np.zeros((5, L, H))
        cnt = np.zeros(5)
        for row in rows:
            o, w2t = run_verse(model, tok, bos, row["text"], row["spans"], device, attn=True)
            A = torch.stack(o.attentions)[:, 0]  # [L, H, T, T]
            for i, ti in enumerate(w2t):
                if not ti:
                    continue
                for k in range(1, 5):
                    if i - k >= 0 and w2t[i - k]:
                        acc[k] += A[:, :, ti[-1], w2t[i - k]].sum(-1).cpu().numpy()
                        cnt[k] += 1
        gen[v] = acc / np.maximum(cnt, 1)[:, None, None]
        print(f"{v}: generic offset attention from {len(rows)} verses")
    prev_score = (gen["web"][1] + gen["jnt"][1]) / 2  # previous-word attention, both varieties

    # marker attention vs offset-matched generic attention
    for f in FEATURES:
        for v in VARIETIES:
            key = f"{f}_{v}"
            m = hz.get(f"marker_{key}")
            if m is None:
                continue
            rows = read_jsonl(DERIVED / f"sample_{v}.jsonl")
            offs = Counter()
            for row in rows:
                for i, y in row["labels"][f].items():
                    mi = marker_index(v, f, row["words"], int(i))
                    if y == 1 and mi is not None:
                        offs[int(i) - mi] += 1
            n = sum(offs.values())
            expect = sum(c * gen[v][k] for k, c in offs.items() if k < 5) / max(1, n)
            top = [tuple(divmod(int(x), H)) for x in np.argsort(-m[:r].ravel())[:5]]
            res["offset"][key] = {
                "offsets": {str(k): c for k, c in sorted(offs.items())},
                "top_heads": [{"head": [l, h], "marker_attn": round(float(m[l, h]), 3),
                               "same_offset_generic": round(float(expect[l, h]), 3),
                               "prev_word_rank": int((prev_score[:r] > prev_score[l, h]).sum()) + 1}
                              for l, h in top]}
            print(f"{key}: " + ", ".join(f"{l}.{h} marker {m[l, h]:.2f} vs generic {expect[l, h]:.2f} "
                                         f"(prev-word rank {int((prev_score[:r] > prev_score[l, h]).sum()) + 1})"
                                         for l, h in top))

    # ablation of the top generic previous-word heads
    Wo = torch.tensor(np.load(out / "weights.npz")["W_O"], dtype=torch.float32, device=device)
    order = [tuple(divmod(int(x), H)) for x in np.argsort(-prev_score[:r].ravel())]
    for f in FEATURES:
        for v in VARIETIES:
            key = f"{f}_{v}"
            if key not in P:
                continue
            d_r = torch.tensor(P[key][r], dtype=torch.float32, device=device)
            rows = [row for row in read_jsonl(DERIVED / f"sample_{v}.jsonl") if row["labels"][f]][:a.abl_n]

            def gap(heads):
                byl = defaultdict(list)
                for l, h in heads:
                    byl[l].append(h)

                def edit(l, z):
                    if l not in byl:
                        return z
                    z = z.clone().view(*z.shape[:-1], H, hd)
                    z[..., byl[l], :] = 0
                    return z.view(*z.shape[:-2], H * hd)
                cap.edit = edit if heads else None
                proj, yy = [], []
                for row in rows:
                    o, w2t = run_verse(model, tok, bos, row["text"], row["spans"], device)
                    hs = o.hidden_states[r][0]
                    for i, y in row["labels"][f].items():
                        if w2t[int(i)]:
                            proj.append(float(hs[w2t[int(i)][-1]] @ d_r))
                            yy.append(y)
                cap.edit = None
                proj, yy = np.array(proj), np.array(yy)
                return proj[yy == 1].mean() - proj[yy == 0].mean()

            g0 = gap([])
            runs = []
            for k in a.k:
                runs.append({"k": k, "heads": [list(x) for x in order[:k]],
                             "retained": round(float(gap(order[:k]) / g0), 3)})
            step4 = {(run["k"]): run["retained"] for run in
                     json.loads(open(f"specimens/bible-pair_{a.model}_heads.json").read())
                     ["features"][f][v].get("ablation", {}).get("runs", []) if run["mode"] == "zero"}
            res["prev_ablation"][key] = {"runs": runs, "step4_zero": {str(k): x for k, x in step4.items()}}
            print(f"{key}: prev-word heads zeroed, retained " +
                  ", ".join(f"k={x['k']} {x['retained']:.2f} (marker heads {step4.get(x['k'])})" for x in runs))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-410m")
    ap.add_argument("--dynamic", action="store_true", help="also run the model-based controls")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n", type=int, default=300, help="verses for generic offset attention")
    ap.add_argument("--abl-n", type=int, default=300)
    ap.add_argument("--k", type=int, nargs="+", default=[5, 20])
    ap.add_argument("--readout", type=int, default=12)
    a = ap.parse_args()
    out = OUT / a.model
    meta = json.loads((out / "meta.json").read_text())
    dest = f"specimens/bible-pair_{a.model}_controls.json"
    t0 = time.time()
    res = {"model": meta["model"], "static": static(a, out, meta)}
    if a.dynamic:
        res["dynamic"] = dynamic(a, out, meta)
    elif os.path.exists(dest):  # keep earlier dynamic results
        old = json.load(open(dest))
        if "dynamic" in old:
            res["dynamic"] = old["dynamic"]
    write_json(dest, res)
    print(f"-> {dest} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
