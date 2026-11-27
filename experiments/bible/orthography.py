"""Step 5 control: does the JNT result come from spelling or from grammar?

Compares the JNT, the respelled JNT ("jntr": English-based spelling, Creole grammar and
markers kept, see respell.py) and the WEB on the same verses and the same labelled words:

  tokenisation   tokens per word and single-token share
  probes         balanced accuracy per layer (5-fold CV by verse) for jntr; JNT and WEB from step 3
  transfer       AUC of jntr words projected onto the WEB and the JNT probe directions
  marker heads   attention from positive words to their marker, for the step-4 JNT marker heads
  ablation       zeroing the step-4 JNT marker heads (top 5 / 20): signal retained at the readout
                 layer for jntr, against random heads, next to the JNT and WEB results

Needs: respell.py, then extract.py --varieties jntr. Writes
specimens/bible-pair_<model>_orthography.json (numbers only).
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from collections import defaultdict

os.environ.setdefault("HF_HOME", os.path.abspath("data/hf_cache"))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from common import DERIVED, FEATURES, OUT, read_jsonl, write_json

warnings.filterwarnings("ignore", category=UserWarning)


def labelled(out, v):
    z = np.load(out / f"acts_{v}.npz")
    rows = read_jsonl(DERIVED / f"sample_{v}.jsonl")
    return z, rows


def xy(z, rows, f):
    idx, y = [], []
    for k, (vi, wi) in enumerate(zip(z["verse"], z["word"])):
        lab = rows[vi]["labels"][f].get(str(wi))
        if lab is not None:
            idx.append(k)
            y.append(lab)
    return np.array(idx), np.array(y)


def cv(X, y, g, C=0.05):
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(X, y, g):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    return round(float(balanced_accuracy_score(y, pred)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-410m")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--abl-n", type=int, default=300)
    ap.add_argument("--random-draws", type=int, default=2)
    ap.add_argument("--no-ablation", action="store_true")
    a = ap.parse_args()
    out = OUT / a.model
    meta = json.loads((out / "meta.json").read_text())
    L, H, hd = meta["n_layers"], meta["n_heads"], meta["d_head"]
    probes = json.loads(open(f"specimens/bible-pair_{a.model}_probes.json").read())
    heads = json.loads(open(f"specimens/bible-pair_{a.model}_heads.json").read())
    P = dict(np.load(out / "probes.npz"))
    r = heads["readout_layer"]
    zr, rr = labelled(out, "jntr")
    zj, rj = labelled(out, "jnt")
    assert (zr["verse"] == zj["verse"]).all() and (zr["word"] == zj["word"]).all(), \
        "jntr and jnt must have the same labelled words"

    res = {"model": meta["model"], "readout_layer": r, "tokenisation": {}, "features": {}}
    for v, z in (("web", np.load(out / "acts_web.npz")), ("jnt", zj), ("jntr", zr)):
        t = z["tokens_per_word"]
        res["tokenisation"][v] = {"tokens_per_word": round(float(t.mean()), 3),
                                  "single_token": round(float((t == 1).mean()), 3)}
    print("tokens/word:", {k: v["tokens_per_word"] for k, v in res["tokenisation"].items()})

    for f in FEATURES:
        idx, y = xy(zr, rr, f)
        if min(y.sum(), (1 - y).sum()) < 20:
            continue
        g = zr["verse"][idx]
        X = zr["X"][idx].astype(np.float32)
        acc = [cv(X[:, l], y, g) for l in range(L + 1)]
        ent = {"acc_jntr": acc, "acc_jnt": probes["features"][f]["jnt"].get("acc"),
               "acc_web": probes["features"][f]["web"].get("acc")}
        for src in ("web", "jnt"):
            if f"{f}_{src}" in P:
                D = P[f"{f}_{src}"]
                ent[f"transfer_auc_from_{src}"] = [round(float(roc_auc_score(y, X[:, l] @ D[l])), 3)
                                                   for l in range(L + 1)]
        ent["transfer_auc_web_to_jnt"] = probes["features"][f]["web"].get("transfer_auc_to_jnt")
        res["features"][f] = ent
        b = int(np.argmax(acc))
        print(f"{f:12s} jntr acc L0 {acc[0]:.3f} best {acc[b]:.3f} (L{b}) | jnt L0 {ent['acc_jnt'][0]} "
              f"best {max(ent['acc_jnt']):.3f} | web->jntr AUC max "
              f"{max(ent.get('transfer_auc_from_web', [0])):.3f} vs web->jnt "
              f"{max(ent['transfer_auc_web_to_jnt'] or [0]):.3f}")

    if not a.no_ablation:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from extract import pick_device
        from heads import Capture, marker_index, run_verse
        device = pick_device(a.device)
        tok = AutoTokenizer.from_pretrained(meta["model"], revision=meta["revision"])
        model = AutoModelForCausalLM.from_pretrained(meta["model"], revision=meta["revision"],
                                                     attn_implementation="eager", dtype=torch.float32)
        model.eval().to(device)
        cap = Capture(model, H, hd)
        bos = tok.eos_token_id
        rng = np.random.default_rng(0)
        pool = [(l, h) for l in range(r) for h in range(H)]
        jw = {row["ref"]: row["words"] for row in rj}
        for f in FEATURES:
            runs4 = heads["features"][f]["jnt"].get("ablation", {}).get("runs", [])
            if f not in res["features"] or not runs4:
                continue
            key = f"{f}_jntr"
            # direction for jntr: fit on jntr at the readout layer
            idx, y = xy(zr, rr, f)
            sc = StandardScaler().fit(zr["X"][idx, r].astype(np.float32))
            clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(
                sc.transform(zr["X"][idx, r].astype(np.float32)), y)
            d = clf.coef_[0] / sc.scale_
            d_r = torch.tensor(d / np.linalg.norm(d), dtype=torch.float32, device=device)
            rows = [row for row in rr if row["labels"][f]][:a.abl_n]

            def gap(hs_):
                byl = defaultdict(list)
                for l, h in hs_:
                    byl[l].append(h)

                def edit(l, z):
                    if l not in byl:
                        return z
                    z = z.clone().view(*z.shape[:-1], H, hd)
                    z[..., byl[l], :] = 0
                    return z.view(*z.shape[:-2], H * hd)
                cap.edit = edit if hs_ else None
                proj, yy, matt, mn = [], [], np.zeros((L, H)), 0
                for row in rows:
                    o, w2t = run_verse(model, tok, bos, row["text"], row["spans"], device, attn=not hs_)
                    hsr = o.hidden_states[r][0]
                    for i, lab in row["labels"][f].items():
                        i = int(i)
                        if not w2t[i]:
                            continue
                        proj.append(float(hsr[w2t[i][-1]] @ d_r))
                        yy.append(lab)
                        # marker position from the original JNT words (same indices)
                        m = marker_index("jnt", f, jw[row["ref"]], i)
                        if not hs_ and lab == 1 and m is not None and w2t[m]:
                            A = torch.stack(o.attentions)[:, 0, :, w2t[i][-1], :]
                            matt += A[:, :, w2t[m]].sum(-1).cpu().numpy()
                            mn += 1
                cap.edit = None
                proj, yy = np.array(proj), np.array(yy)
                return proj[yy == 1].mean() - proj[yy == 0].mean(), (matt / max(mn, 1) if mn else None)

            g0, matt = gap([])
            mj = np.load(out / "heads.npz")[f"marker_{f}_jnt"]
            top5 = [tuple(x) for x in runs4[0]["heads"][:5]]
            ent = {"marker_attn_top5": [{"head": list(hh), "jnt": round(float(mj[hh]), 3),
                                         "jntr": round(float(matt[hh]), 3) if matt is not None else None}
                                        for hh in top5], "runs": []}
            for run in runs4:
                if run["mode"] != "zero":
                    continue
                hs_ = [tuple(x) for x in run["heads"]]
                k = len(hs_)
                ret = gap(hs_)[0] / g0
                rnd = [gap([pool[i] for i in rng.choice(len(pool), k, replace=False)])[0] / g0
                       for _ in range(a.random_draws)]
                web = next((x for x in heads["features"][f]["web"].get("ablation", {}).get("runs", [])
                            if x["mode"] == "zero" and x["k"] == k), {})
                ent["runs"].append({"k": k, "jntr": round(float(ret), 3),
                                    "jntr_random": [round(float(x), 3) for x in rnd],
                                    "jnt": run["retained"], "web": web.get("retained")})
                print(f"{key} k={k}: retained jntr {ret:.2f} (random {np.mean(rnd):.2f}) | "
                      f"jnt {run['retained']:.2f} | web {web.get('retained')}")
            print(f"{key} marker attention top-5 JNT heads: " +
                  ", ".join(f"{e['head'][0]}.{e['head'][1]} jnt {e['jnt']:.2f} jntr {e['jntr']:.2f}"
                            for e in ent["marker_attn_top5"]))
            res["features"][f]["ablation"] = ent

    dest = f"specimens/bible-pair_{a.model}_orthography.json"
    write_json(dest, res)
    print("->", dest)


if __name__ == "__main__":
    main()
