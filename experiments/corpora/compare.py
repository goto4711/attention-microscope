"""Does the Standard-English feature direction still see the feature in each collection?

For every collection with a sample (data/derived/sample_<key>.jsonl) and activations
(experiments/out/<model>/acts_<key>.npz), and each feature (plural, past, progressive):

  transfer   ROC AUC, per layer, of the collection's labelled words projected onto the
             WEB probe direction (step 3). 1 = the standard direction separates the
             feature perfectly here; 0.5 = it does not see it at all
  in-house   balanced accuracy of a probe trained and tested within the collection
             (5-fold CV by sentence): is the feature linearly there at all?
  gap        in-house accuracy minus what the WEB direction achieves: the feature is
             encoded, but not where the standard puts it
  WEB itself held-out transfer (direction fitted on other folds), so it is not in-sample

  unmarked   once hand labels are imported (variety_xlsx.py): the same AUC for the
             variety's unmarked forms (plural without -s, past without -ed, progressive
             without be) next to its standard-marked forms, against the same negatives.
             This is the test that does not depend on the standard tagger

plus tokens per word, label counts and sources, and the collection's distance from
Standard English. Summary: transfer AUC averaged over layers 8-16, and its Spearman
correlation with distance across collections.

Writes specimens/corpora_<model>.json (numbers only).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bible"))
sys.path.insert(0, str(HERE))
from common import DERIVED, FEATURES, OUT, read_jsonl, write_json  # noqa: E402
from tokenisation_profile import DISTANCE, LABEL  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
ORDER = ["web", "ice-ja-written", "aapb-howard", "aapb-eotp", "ice-in-written", "ice-in-spoken",
         "ice-ja-spoken", "wpa", "coraal", "lince", "jntr", "jnt"]
EXTRA = {"jntr": ("Jamaican NT, respelled", 0.89)}
MID = slice(8, 17)


def jnt_unmarked(f):
    """JNT hand labels the rules left unlabelled (auto empty): forms without an overt marker."""
    import csv
    p = Path("data/annotation") / f"jnt_{f}.csv"
    if not p.exists():
        return set()
    return {(r["ref"], int(r["word_idx"])) for r in csv.DictReader(open(p, encoding="utf-8"))
            if r["gold"].strip() == "1" and not r["auto"].strip()}


def xy(z, rows, f, key=None):
    idx, y, src, unm = [], [], [], []
    ju = jnt_unmarked(f) if key == "jnt" else set()
    for k, (vi, wi) in enumerate(zip(z["verse"], z["word"])):
        lab = rows[vi]["labels"][f].get(str(wi))
        if lab is not None:
            idx.append(k)
            y.append(lab)
            src.append(rows[vi].get("sources", {}).get(f, {}).get(str(wi), "?"))
            unm.append(rows[vi].get("marking", {}).get(f, {}).get(str(wi)) == "unmarked"
                       or (rows[vi]["ref"], int(wi)) in ju)
    return np.array(idx, int), np.array(y, int), src, np.array(unm, bool)


def boot_ci(proj, y, groups, n=500, seed=0):
    """95% interval of the mean AUC over the columns of proj, resampling sentences (clusters)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.choice(ug, len(ug))])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        vals.append(np.mean([roc_auc_score(yy, proj[idx, c]) for c in range(proj.shape[1])]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def fit(X, y, C=0.05):
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-410m")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 12, 16, 20, 24])
    a = ap.parse_args()
    out = OUT / a.model
    P = dict(np.load(out / "probes.npz"))
    res = {"model": a.model, "layers_inhouse": a.layers, "mid_layers": [MID.start, MID.stop - 1],
           "collections": {}, "summary": {}}
    for key in ORDER:
        za, sp = out / f"acts_{key}.npz", DERIVED / f"sample_{key}.jsonl"
        if not (za.exists() and sp.exists()):
            continue
        z, rows = np.load(za), read_jsonl(sp)
        label, dist = EXTRA.get(key, (LABEL.get(key, key), DISTANCE.get(key)))
        t = z["tokens_per_word"]
        ent = {"label": label, "distance": dist, "sentences": len(rows),
               "tokens_per_word": round(float(t.mean()), 3), "features": {}}
        for f in FEATURES:
            idx, y, src, unm = xy(z, rows, f, key)
            fe = {"n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
                  "sources": {s: src.count(s) for s in set(src)}}
            ent["features"][f] = fe
            if min(fe["n_pos"], fe["n_neg"]) < 15:
                fe["skipped"] = "fewer than 15 examples in a class"
                continue
            # hand-labelled unmarked positives are tested separately (below); the main
            # transfer and in-house figures use the standard-marked labels only, so that
            # collections with and without hand annotation stay comparable
            allX, ally = z["X"][idx].astype(np.float32), y
            keep = ~(unm & (y == 1))
            X, y, g = allX[keep], y[keep], z["verse"][idx][keep]
            D = P[f"{f}_web"]
            if key == "web":  # held-out: direction fitted on other folds
                proj = np.zeros((len(y), X.shape[1]))
                for l in range(X.shape[1]):
                    for tr, te in GroupKFold(5).split(X, y, g):
                        sc, clf = fit(X[tr, l], y[tr])
                        proj[te, l] = X[te, l] @ (clf.coef_[0] / sc.scale_)
            else:
                proj = np.einsum("nld,ld->nl", X, D)
            auc = [roc_auc_score(y, proj[:, l]) for l in range(X.shape[1])]
            fe["transfer_mid_ci"] = boot_ci(proj[:, MID], y, g)
            fe["transfer_auc"] = [round(float(x), 3) for x in auc]
            inh = []
            for l in a.layers:
                pred = np.zeros_like(y)
                for tr, te in GroupKFold(5).split(X, y, g):
                    sc, clf = fit(X[tr, l], y[tr])
                    pred[te] = clf.predict(sc.transform(X[te, l]))
                inh.append(round(float(balanced_accuracy_score(y, pred)), 3))
            fe["inhouse_acc"] = inh
            fe["transfer_mid"] = round(float(np.mean(auc[MID])), 3)
            # hand-labelled unmarked forms (plural without -s, past without -ed, progressive
            # without be) against the same negatives, next to the standard-marked positives
            pos_u = unm & (ally == 1)
            if pos_u.sum() >= 10 and key != "web":
                y, gg = ally, z["verse"][idx]
                pm = np.einsum("nld,ld->nl", allX[:, MID], D[MID])
                neg, pos_m = y == 0, (y == 1) & ~unm
                def mid_auc(pos):
                    sel = pos | neg
                    return (round(float(np.mean([roc_auc_score(y[sel], pm[sel, c]) for c in range(pm.shape[1])])), 3),
                            boot_ci(pm[sel], y[sel], gg[sel]))
                (u, uci), m = mid_auc(pos_u), (mid_auc(pos_m) if pos_m.sum() >= 10 else (None, None))
                fe["unmarked"] = {"n_pos_unmarked": int(pos_u.sum()), "transfer_mid_unmarked": u,
                                  "transfer_mid_unmarked_ci": uci, "transfer_mid_marked": m[0] if m else None,
                                  "transfer_mid_marked_ci": m[1] if m else None}
        res["collections"][key] = ent
        print(f"{label:34s} {dist if dist is not None else '-':>5} tok/w {ent['tokens_per_word']:.2f} | " +
              " | ".join(f"{f[:4]} {fe.get('transfer_mid', '-')} (in {fe.get('inhouse_acc', [None]*4)[3]}, "
                         f"n {fe['n_pos']}/{fe['n_neg']})" for f, fe in ent["features"].items()))

    for f in FEATURES:
        pts = [(c["distance"], c["features"][f]["transfer_mid"]) for k, c in res["collections"].items()
               if k != "jntr" and c["distance"] is not None and "transfer_mid" in c["features"][f]]
        if len(pts) >= 4:
            rho, p = spearmanr([d for d, _ in pts], [v for _, v in pts])
            res["summary"][f] = {"n": len(pts), "spearman_rho": round(float(rho), 3), "p": round(float(p), 4)}
            print(f"{f}: Spearman rho(distance, transfer AUC layers 8-16) = {rho:.2f} (p={p:.3f}, n={len(pts)})")
    for key, c in res["collections"].items():
        for f, fe in c["features"].items():
            if "unmarked" in fe:
                u = fe["unmarked"]
                print(f"{c['label']:34s} {f:12s} WEB direction, layers 8-16: standard-marked "
                      f"{u['transfer_mid_marked']} {u['transfer_mid_marked_ci']} vs unmarked "
                      f"{u['transfer_mid_unmarked']} {u['transfer_mid_unmarked_ci']} "
                      f"(n unmarked {u['n_pos_unmarked']})")
    write_json(f"specimens/corpora_{a.model}.json", res)
    print(f"-> specimens/corpora_{a.model}.json")


if __name__ == "__main__":
    main()
