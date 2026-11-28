"""Step 6: when, during training, do the heads behind steps 4-5 appear?

For a series of Pythia training checkpoints (revisions step0 ... step143000) and the first
--n verses of the WEB and JNT samples, per checkpoint:

  prev-word     per head, attention from each word to the word just before it; the
                strongest head, and how many heads give it more than half their attention
  be->ing       per head, attention from a WEB progressive -ing verb to its form of be;
                head 4.5 (the final model's progressive head) and the best head
  jnt marker    per head, attention from a JNT past / progressive verb to its marker
                (did, en, a ...)
  probes        balanced accuracy of per-variety probes at layers 4, 8 and 12 (5-fold CV by
                verse), and the AUC of the WEB direction on the JNT words (transfer)
  ablation      at layer 12, the gap between positive and negative words along this
                checkpoint's own probe direction, and the share left when zeroing this
                checkpoint's 20 strongest previous-word heads (layers < 12) against 20
                random heads; for the WEB progressive, also its 5 strongest be->ing heads

Downloads come from the Hugging Face hub (set HF_HUB_OFFLINE=1 once they are cached).

  python experiments/bible/checkpoints.py [--steps 0 512 1000 ...] [--n 250]

The verses are the first --n plus all verses with a progressive in either variety.

Writes specimens/bible-pair_<model>_checkpoints.json (numbers only), one entry per
checkpoint, saved after each so a long run can be stopped and resumed (--skip-done).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.abspath("data/hf_cache"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from common import DERIVED, FEATURES, read_jsonl, write_json  # noqa: E402
from extract import pick_device  # noqa: E402
from heads import Capture, cv_decode, marker_index, run_verse  # noqa: E402

STEPS = [0, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]
PROBE_LAYERS = [4, 8, 12]
READOUT = 12
K = 20


def collect(model, tok, bos, rows, variety, device, L, H):
    """One pass per verse: attention statistics and hidden states of labelled words."""
    prev, prev_n = np.zeros((L, H)), 0
    mk = {f: [np.zeros((L, H)), 0] for f in ("past", "progressive")}
    words = []  # (verse, word, {f: y}, hidden [len(PROBE_LAYERS)+1, d])
    layers = PROBE_LAYERS + ([READOUT] if READOUT not in PROBE_LAYERS else [])
    for vi, row in enumerate(rows):
        o, w2t = run_verse(model, tok, bos, row["text"], row["spans"], device, attn=True)
        A = torch.stack(o.attentions)[:, 0].float().cpu().numpy()  # [L, H, T, T]
        hs = torch.stack([o.hidden_states[l][0] for l in layers]).float().cpu().numpy()
        for i in range(1, len(w2t)):
            if w2t[i] and w2t[i - 1]:
                prev += A[:, :, w2t[i][-1], w2t[i - 1]].sum(-1)
                prev_n += 1
        for f in mk:
            for i, y in row["labels"][f].items():
                i = int(i)
                if y != 1 or not w2t[i]:
                    continue
                m = marker_index(variety, f, row["words"], i)
                if m is not None and w2t[m]:
                    mk[f][0] += A[:, :, w2t[i][-1], w2t[m]].sum(-1)
                    mk[f][1] += 1
        lab = defaultdict(dict)
        for f in FEATURES:
            for i, y in row["labels"][f].items():
                lab[int(i)][f] = y
        for i, ys in lab.items():
            if w2t[i]:
                words.append((vi, i, ys, hs[:, w2t[i][-1]]))
    out = {"prev": prev / max(prev_n, 1),
           "marker": {f: (v / n if n else None, n) for f, (v, n) in mk.items()}}
    return out, words, layers


def xy(words, f, li):
    sel = [(v, h[li], ys[f]) for v, _, ys, h in words if f in ys]
    if not sel:
        return None, None, None
    g, X, y = zip(*sel)
    return np.stack(X), np.array(y), np.array(g)


def direction(X, y):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
    w = clf.coef_[0] / sc.scale_
    return w / np.linalg.norm(w)


def gap(model, tok, bos, rows, f, w, heads, cap, device, H, hd):
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
    wt = torch.tensor(w, dtype=torch.float32, device=device)
    pos, neg = [], []
    for row in rows:
        lab = row["labels"][f]
        if not lab:
            continue
        o, w2t = run_verse(model, tok, bos, row["text"], row["spans"], device)
        hs = o.hidden_states[READOUT][0]
        for i, y in lab.items():
            if w2t[int(i)]:
                (pos if y == 1 else neg).append(float(hs[w2t[int(i)][-1]] @ wt))
    cap.edit = None
    return float(np.mean(pos) - np.mean(neg)) if pos and neg else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--steps", type=int, nargs="+", default=STEPS)
    ap.add_argument("--n", type=int, default=250, help="verses per variety")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    tag = a.model.split("/")[-1]
    path = Path(f"specimens/bible-pair_{tag}_checkpoints.json")
    res = json.loads(path.read_text()) if (a.skip_done and path.exists()) else {
        "model": a.model, "n_verses": a.n, "readout_layer": READOUT, "probe_layers": PROBE_LAYERS,
        "k": K, "checkpoints": {}}
    device = pick_device(a.device)
    tok = AutoTokenizer.from_pretrained(a.model)
    bos = tok.eos_token_id
    full = {v: read_jsonl(DERIVED / f"sample_{v}.jsonl") for v in ("web", "jnt")}
    # the first n verses, plus every verse with a progressive (rare in the WEB), so that
    # be->ing and the progressive ablation have enough examples; the samples are aligned
    keep = sorted(set(range(a.n)) | {i for v in full for i, r in enumerate(full[v])
                                     if 1 in r["labels"]["progressive"].values()})
    rows = {v: [full[v][i] for i in keep] for v in full}
    print(f"{len(keep)} verses per variety", flush=True)
    rng = np.random.default_rng(0)
    for step in a.steps:
        rev = "main" if step == 143000 else f"step{step}"
        if str(step) in res["checkpoints"]:
            continue
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(a.model, revision=rev, attn_implementation="eager",
                                                     dtype=torch.float32).eval().to(device)
        L, H = model.config.num_hidden_layers, model.config.num_attention_heads
        hd = model.config.hidden_size // H
        cap = Capture(model, H, hd)
        stats, words = {}, {}
        for v in rows:
            stats[v], words[v], layers = collect(model, tok, bos, rows[v], v, device, L, H)
        prev = (stats["web"]["prev"] + stats["jnt"]["prev"]) / 2
        order = [tuple(divmod(int(x), H)) for x in np.argsort(-prev[:READOUT].ravel())]
        ent = {"revision": rev,
               "prev_word": {"max": round(float(prev.max()), 3),
                             "argmax": list(divmod(int(prev.argmax()), H)),
                             "n_over_half": int((prev > 0.5).sum()),
                             "top": [[l, h, round(float(prev[l, h]), 3)] for l, h in order[:5]],
                             "head_4_5": round(float(prev[4, 5]), 3)},
               "marker": {}, "probes": {}, "ablation": {}}
        bi, n_bi = stats["web"]["marker"]["progressive"]
        if bi is not None:
            ent["be_ing"] = {"n": n_bi, "head_4_5": round(float(bi[4, 5]), 3),
                             "max": round(float(bi.max()), 3), "argmax": list(divmod(int(bi.argmax()), H))}
        for f in ("past", "progressive"):
            m, n = stats["jnt"]["marker"][f]
            if m is not None:
                ent["marker"][f"jnt_{f}"] = {"n": n, "max": round(float(m.max()), 3),
                                             "argmax": list(divmod(int(m.argmax()), H)),
                                             "head_4_5": round(float(m[4, 5]), 3)}
        for f in FEATURES:
            pe = {}
            for li, l in enumerate(PROBE_LAYERS):
                for v in rows:
                    X, y, g = xy(words[v], f, li)
                    if X is not None:
                        acc = cv_decode(X, y, g)
                        pe.setdefault(v, []).append(None if np.isnan(acc) else round(float(acc), 3))
                Xw, yw, _ = xy(words["web"], f, li)
                Xj, yj, _ = xy(words["jnt"], f, li)
                if Xw is not None and Xj is not None and min(yw.sum(), (1 - yw).sum()) >= 10 \
                        and 0 < yj.sum() < len(yj):
                    pe.setdefault("transfer", []).append(round(float(roc_auc_score(yj, Xj @ direction(Xw, yw))), 3))
            ent["probes"][f] = pe
        ri = layers.index(READOUT)
        pool = [(l, h) for l in range(READOUT) for h in range(H)]
        todo = [("jnt", "past"), ("jnt", "progressive"), ("web", "progressive"), ("web", "past")]
        for v, f in todo:
            X, y, _ = xy(words[v], f, ri)
            if X is None or min(y.sum(), (1 - y).sum()) < 10:
                continue
            w = direction(X, y)
            g0 = gap(model, tok, bos, rows[v], f, w, [], cap, device, H, hd)
            if not g0:
                continue
            ab = {"n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
                  "prev_word": round(gap(model, tok, bos, rows[v], f, w, order[:K], cap, device, H, hd) / g0, 3),
                  "random": round(gap(model, tok, bos, rows[v], f, w,
                                      [pool[i] for i in rng.choice(len(pool), K, replace=False)],
                                      cap, device, H, hd) / g0, 3)}
            if v == "web" and f == "progressive" and bi is not None:
                bo = [tuple(divmod(int(x), H)) for x in np.argsort(-bi[:READOUT].ravel())[:5]]
                ab["be_ing_top5"] = round(gap(model, tok, bos, rows[v], f, w, bo, cap, device, H, hd) / g0, 3)
                ab["be_ing_heads"] = [list(x) for x in bo]
            ent["ablation"][f"{v}_{f}"] = ab
        res["checkpoints"][str(step)] = ent
        write_json(str(path), res)
        p8 = {f: ent["probes"][f].get("jnt", [None] * 3)[1] for f in FEATURES}
        print(f"step {step:6d}: prev-word max {ent['prev_word']['max']} ({ent['prev_word']['n_over_half']} heads > .5), "
              f"4.5 be->ing {ent.get('be_ing', {}).get('head_4_5')} | JNT probes L8 {p8} | ablation "
              + ", ".join(f"{k} prev {v['prev_word']} rand {v['random']}" for k, v in ent["ablation"].items())
              + f" | {time.time() - t0:.0f}s", flush=True)
        for hnd in cap.handles:
            hnd.remove()
        del model
    print(f"-> {path}")


if __name__ == "__main__":
    main()
