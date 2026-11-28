"""Step 3: linear probes per feature, per variety, on the residual stream entering each layer.

For each feature f, variety v and layer l:
  acc        balanced accuracy of a logistic-regression probe, 5-fold cross-validation
             grouped by verse (no verse appears in both train and test)
  direction  unit vector in residual-stream space (probe weights mapped back through
             the standardisation), fitted on all of v's labelled words: the real
             counterpart of a feature row in the toy page
Across varieties:
  transfer   ROC AUC of the other variety's words projected onto this variety's
             direction (threshold-free, so a shift in residual norm between the
             varieties does not count as failure)
  cosine     between the WEB and JNT directions at each layer

Writes directions to experiments/out/<model>/probes.npz and a text-free summary to
specimens/bible-pair_<model>_probes.json.
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from common import DERIVED, FEATURES, OUT, VARIETIES, read_jsonl, write_json

warnings.filterwarnings("ignore", category=UserWarning)
MIN_CLASS = 20


def load(out, variety):
    z = np.load(out / f"acts_{variety}.npz")
    rows = read_jsonl(DERIVED / f"sample_{variety}.jsonl")
    lab = {f: [] for f in FEATURES}
    for k, (v, w) in enumerate(zip(z["verse"], z["word"])):
        for f in FEATURES:
            y = rows[v]["labels"][f].get(str(w))
            if y is not None:
                lab[f].append((k, y, rows[v]["sources"][f][str(w)]))
    return z["X"], z["verse"], lab


def fit(X, y, C):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
    return sc, clf


def direction(sc, clf):
    w = clf.coef_[0] / sc.scale_
    return w / np.linalg.norm(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-410m", help="folder name under experiments/out")
    ap.add_argument("--C", type=float, default=0.05, help="inverse L2 strength")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    out = OUT / a.model
    meta = json.loads((out / "meta.json").read_text())
    Lp1 = meta["n_layers"] + 1

    data = {v: load(out, v) for v in VARIETIES}
    dirs = {}
    summary = {"model": meta["model"], "revision": meta["revision"],
               "unit": "word (last subtoken)", "layers": list(range(Lp1)),
               "layer_note": "layer l = residual stream entering layer l; last = after final layer",
               "probe": f"L2 logistic regression, C={a.C}, balanced classes, "
                        f"{a.folds}-fold CV grouped by verse",
               "tokens_per_word": {v: meta["varieties"][v]["tokens_per_word_mean"] for v in VARIETIES},
               "features": {}}

    for f in FEATURES:
        fs = summary["features"][f] = {}
        for v in VARIETIES:
            X, verse, lab = data[v]
            idx = np.array([k for k, _, _ in lab[f]], int)
            y = np.array([y for _, y, _ in lab[f]], int)
            n1, n0 = int(y.sum()), int((1 - y).sum())
            entry = {"n_pos": n1, "n_neg": n0,
                     "label_sources": dict(Counter(s for _, _, s in lab[f]))}
            fs[v] = entry
            if min(n0, n1) < MIN_CLASS:
                entry["skipped"] = f"fewer than {MIN_CLASS} examples in a class"
                print(f"{f:12s} {v}: skipped ({n1} pos / {n0} neg)")
                continue
            groups = verse[idx]
            acc, D = [], np.zeros((Lp1, X.shape[-1]), np.float32)
            for l in range(Lp1):
                Xl = X[idx, l].astype(np.float32)
                pred = np.zeros_like(y)
                for tr, te in GroupKFold(a.folds).split(Xl, y, groups):
                    sc, clf = fit(Xl[tr], y[tr], a.C)
                    pred[te] = clf.predict(sc.transform(Xl[te]))
                acc.append(balanced_accuracy_score(y, pred))
                D[l] = direction(*fit(Xl, y, a.C))
            entry["acc"] = [round(float(x), 3) for x in acc]
            dirs[(f, v)] = D
            best = int(np.argmax(acc))
            print(f"{f:12s} {v}: {n1} pos / {n0} neg, best layer {best} acc {acc[best]:.3f}, "
                  f"layer 0 {acc[0]:.3f}, final {acc[-1]:.3f}")

        # cross-variety transfer and direction similarity
        for src, tgt in (("web", "jnt"), ("jnt", "web")):
            if (f, src) not in dirs or "skipped" in fs[tgt]:
                continue
            X, _, lab = data[tgt]
            idx = np.array([k for k, _, _ in lab[f]], int)
            y = np.array([y for _, y, _ in lab[f]], int)
            auc = [roc_auc_score(y, X[idx, l].astype(np.float32) @ dirs[(f, src)][l])
                   for l in range(Lp1)]
            fs[src][f"transfer_auc_to_{tgt}"] = [round(float(x), 3) for x in auc]
            print(f"{f:12s} {src}->{tgt}: best transfer AUC {max(auc):.3f} "
                  f"at layer {int(np.argmax(auc))}")
        if (f, "web") in dirs and (f, "jnt") in dirs:
            cos = (dirs[(f, "web")] * dirs[(f, "jnt")]).sum(1)
            fs["cosine_web_jnt"] = [round(float(x), 3) for x in cos]

    np.savez(out / "probes.npz", **{f"{f}_{v}": D for (f, v), D in dirs.items()})
    dest = f"specimens/bible-pair_{a.model}_probes.json"
    write_json(dest, summary)
    print(f"-> {out / 'probes.npz'}, {dest}")


if __name__ == "__main__":
    main()
