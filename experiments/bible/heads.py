"""Step 4: per head and per variety, what does each attention head see of each feature?

For every layer l, head h, feature f and variety v (WEB, JNT):

  visibility   ‖W_X · LN(d)‖ for X in Q, K, V, where d is the step-3 probe direction for
               (f, v) at layer l and LN(d) = ln_g ⊙ (d - mean(d)) is the direction as the
               head sees it after the input LayerNorm (centring and gain). Reported as a
               ratio to the same norm for random unit directions: 1 = as visible as an
               arbitrary direction, < 1 = the lens dims the feature, > 1 = it stands out.
  marker_attn  mean attention from labelled positive words to the word carrying the overt
               marker (JNT: did/en/ben/neehn/a/modal before the verb, the noun before plural
               "dem"; WEB: the form of "be" before a progressive). A Clark et al. (2019)
               style score; no marker -> not computed (WEB plural and past are inflectional).
  head_decode  balanced accuracy of a probe on the head's output z_h (d_head = 64, before
               W_O) at the labelled word, 5-fold CV grouped by verse: is the feature
               decodable from what this head writes?
  ablation     causal check. Pick the K heads (layers < readout) with the highest
               marker_attn (when there are at least 20 marker examples; otherwise the highest
               head_decode) and remove, from their summed contribution to the residual
               stream, the component along the probe direction d_r at the readout layer r
               ("direction"), or silence them entirely ("zero"). Measure the gap between the
               mean projection onto d_r of positive and negative words at layer r, and report
               the share of the clean gap that survives ("retained"; 1 = no effect), against
               the same ablation on K random heads from the same layers. (AUC is also given
               but saturates: d_r was fitted on these words.)

Reads experiments/out/<model>/{weights,probes}.npz, meta.json and the sample files.
Writes experiments/out/<model>/heads.npz (all arrays) and a text-free summary
specimens/bible-pair_<model>_heads.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

os.environ.setdefault("HF_HOME", os.path.abspath("data/hf_cache"))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import DERIVED, FEATURES, OUT, VARIETIES, read_jsonl, write_json
from extract import pick_device, word_token_map

warnings.filterwarnings("ignore", category=UserWarning)
JNT_MARKERS = {"did", "en", "ben", "neehn", "a", "wi", "a-go", "kyan", "kyaan", "mos",
               "shuda", "wuda", "kuda"}
SKIP_ADVERBS = {"no", "jos", "stil", "aalwiez"}
BE_FORMS = {"is", "are", "was", "were", "be", "been", "am", "being"}


# --- static: visibility ----------------------------------------------------
def visibility(W, dirs, n_random=256, seed=0):
    """W: dict of [L, H, hd, d]; dirs: [L+1, d] probe directions. Returns {X: [L, H]}."""
    L = W["W_Q"].shape[0]
    g = W["ln_g"].astype(np.float32)  # [L, d]
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((n_random, g.shape[1])).astype(np.float32)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    out = {}
    for X in "QKV":
        M = W[f"W_{X}"].astype(np.float32)  # [L, H, hd, d]
        vis = np.zeros(M.shape[:2], np.float32)
        for l in range(L):
            d = dirs[l] - dirs[l].mean()
            d = g[l] * d / np.linalg.norm(d)
            r = g[l] * (R - R.mean(1, keepdims=True))
            r /= np.linalg.norm(R - R.mean(1, keepdims=True), axis=1, keepdims=True)
            num = np.linalg.norm(M[l] @ d, axis=-1)                      # [H]
            den = np.linalg.norm(np.einsum("hkd,nd->hnk", M[l], r), axis=-1).mean(1)
            vis[l] = num / den
        out[X] = vis
    return out


# --- markers ----------------------------------------------------------------
def marker_index(variety, feature, words, i):
    lw = [w.lower() for w in words]
    if variety == "jnt":
        if feature == "plural":
            return i - 1 if i >= 1 and lw[i] == "dem" else None
        j = i - 1
        if j >= 0 and lw[j] in SKIP_ADVERBS:
            j -= 1
        return j if j >= 0 and lw[j] in JNT_MARKERS else None
    if feature == "progressive":
        for j in range(i - 1, max(-1, i - 4), -1):
            if lw[j] in BE_FORMS:
                return j
    return None


# --- dynamic pass with hooks ------------------------------------------------
class Capture:
    """Pre-hooks on each layer's attention.dense: capture z (all heads), optionally edit it."""

    def __init__(self, model, H, hd):
        self.H, self.hd = H, hd
        self.z = {}
        self.edit = None  # callable(layer, z) -> z, applied to dense input
        self.post = None  # callable(layer, z, y) -> y, applied to dense output
        self.handles = []
        for l, layer in enumerate(model.gpt_neox.layers):
            self.handles.append(layer.attention.dense.register_forward_pre_hook(self._pre(l)))
            self.handles.append(layer.attention.dense.register_forward_hook(self._post(l)))

    def _pre(self, l):
        def hook(mod, args):
            z = args[0]
            if self.edit is not None:
                z = self.edit(l, z)
            self.z[l] = z
            return (z,)
        return hook

    def _post(self, l):
        def hook(mod, args, y):
            return self.post(l, self.z[l], y) if self.post is not None else y
        return hook


def run_verse(model, tok, bos, text, spans, device, attn=False):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = [bos] + enc["input_ids"]
    offsets = [(0, 0)] + [tuple(o) for o in enc["offset_mapping"]]
    w2t = word_token_map(offsets, spans)
    with torch.no_grad():
        o = model(torch.tensor([ids], device=device), output_hidden_states=True,
                  output_attentions=attn)
    return o, w2t


def labelled(row, f):
    return [(int(i), y) for i, y in row["labels"][f].items()]


def cv_decode(Z, y, groups, C=0.1, folds=5):
    if min(y.sum(), (1 - y).sum()) < 20:
        return np.nan
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(folds).split(Z, y, groups):
        sc = StandardScaler().fit(Z[tr])
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(Z[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(Z[te]))
    return balanced_accuracy_score(y, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="verses for the dynamic pass (0 = all)")
    ap.add_argument("--readout", type=int, default=12, help="layer at which ablation effects are read")
    ap.add_argument("--k", type=int, nargs="+", default=[5, 20], help="heads ablated together")
    ap.add_argument("--random-draws", type=int, default=2)
    ap.add_argument("--abl-n", type=int, default=300, help="verses per ablation run")
    a = ap.parse_args()

    device = pick_device(a.device)
    tag = a.model.split("/")[-1] + ("" if a.revision == "main" else f"@{a.revision}")
    out = OUT / tag
    meta = json.loads((out / "meta.json").read_text())
    L, H, hd = meta["n_layers"], meta["n_heads"], meta["d_head"]
    W = dict(np.load(out / "weights.npz"))
    P = dict(np.load(out / "probes.npz"))
    rng = np.random.default_rng(0)
    r = a.readout

    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision,
                                                 attn_implementation="eager", dtype=torch.float32)
    model.eval().to(device)
    bos = tok.eos_token_id
    cap = Capture(model, H, hd)
    Wo = torch.tensor(W["W_O"], dtype=torch.float32, device=device)  # [L, H, d, hd]

    summary = {"model": meta["model"], "revision": meta["revision"], "n_layers": L,
               "n_heads": H, "readout_layer": r, "k": a.k,
               "visibility_note": "ratio to random unit directions after the input LayerNorm",
               "features": {}}
    arrays = {}

    for f in FEATURES:
        sf = summary["features"][f] = {}
        for v in VARIETIES:
            key = f"{f}_{v}"
            if key not in P:
                sf[v] = {"skipped": "no probe direction (too few labels)"}
                continue
            t0 = time.time()
            ent = sf[v] = {}
            vis = visibility(W, P[key])
            for X in "QKV":
                arrays[f"vis{X}_{key}"] = vis[X]
                ent[f"visibility_{X}"] = np.round(vis[X], 2).tolist()

            rows = read_jsonl(DERIVED / f"sample_{v}.jsonl")
            if a.limit:
                rows = rows[:a.limit]
            rows = [(vi, row) for vi, row in enumerate(rows) if row["labels"][f]]
            Z, ys, groups = [], [], []
            matt, mcount = np.zeros((L, H)), 0
            for vi, row in rows:
                o, w2t = run_verse(model, tok, bos, row["text"], row["spans"], device, attn=True)
                for i, y in labelled(row, f):
                    if not w2t[i]:
                        continue
                    t = w2t[i][-1]
                    Z.append(torch.stack([cap.z[l][0, t].view(H, hd) for l in range(L)]).cpu().numpy())
                    ys.append(y)
                    groups.append(vi)
                    m = marker_index(v, f, row["words"], i)
                    if y == 1 and m is not None and w2t[m]:
                        A = torch.stack(o.attentions)[:, 0, :, t, :]  # [L, H, T]
                        matt += A[:, :, w2t[m]].sum(-1).cpu().numpy()
                        mcount += 1
            Z, ys, groups = np.stack(Z), np.array(ys), np.array(groups)
            dec = np.full((L, H), np.nan)
            for l in range(L):
                for h in range(H):
                    dec[l, h] = cv_decode(Z[:, l, h], ys, groups)
            arrays[f"decode_{key}"] = dec
            ent["n_words"], ent["n_pos"] = int(len(ys)), int(ys.sum())
            ent["head_decode"] = np.round(dec, 3).tolist()
            if mcount:
                matt /= mcount
                arrays[f"marker_{key}"] = matt
                ent["marker_attn"] = np.round(matt, 3).tolist()
                ent["marker_n"] = mcount
                flat = np.argsort(-matt[:r].ravel())[:5]
                ent["top_marker_heads"] = [[int(x // H), int(x % H), round(float(matt.ravel()[x]), 3)]
                                           for x in flat]
            best = np.nanmax(dec) if np.isfinite(dec).any() else float("nan")
            print(f"{key}: {len(ys)} words ({int(ys.sum())} pos), marker examples {mcount}, "
                  f"best head decode {best:.3f}, {time.time() - t0:.0f}s")

            # --- ablation --------------------------------------------------
            d_r = torch.tensor(P[key][r], dtype=torch.float32, device=device)
            if mcount >= 20:
                score, crit = matt[:r], "marker_attn"
            elif np.isfinite(dec).any():
                score, crit = np.nan_to_num(dec[:r], nan=-1), "head_decode"
            else:
                ent["ablation"] = {"skipped": "neither marker attention nor head decoding available"}
                print("  ablation skipped")
                continue
            order = [tuple(divmod(int(x), H)) for x in np.argsort(-score.ravel())]
            abl_rows = rows[:a.abl_n]

            def readout_auc(heads, mode):
                heads_by_l = {}
                for l, h in heads:
                    heads_by_l.setdefault(l, []).append(h)
                if mode == "zero":
                    def edit(l, z):
                        if l not in heads_by_l:
                            return z
                        z = z.clone().view(*z.shape[:-1], H, hd)
                        z[..., heads_by_l[l], :] = 0
                        return z.view(*z.shape[:-2], H * hd)
                    cap.edit, cap.post = edit, None
                elif mode == "direction":
                    def post(l, z, y):
                        if l not in heads_by_l:
                            return y
                        zz = z.view(*z.shape[:-1], H, hd)
                        contrib = sum(zz[..., h, :] @ (Wo[l, h].T @ d_r) for h in heads_by_l[l])
                        return y - contrib.unsqueeze(-1) * d_r
                    cap.edit, cap.post = None, post
                else:
                    cap.edit = cap.post = None
                proj, yy = [], []  # noqa
                for vi, row in abl_rows:
                    o, w2t = run_verse(model, tok, bos, row["text"], row["spans"], device)
                    hs = o.hidden_states[r][0]
                    for i, y in labelled(row, f):
                        if w2t[i]:
                            proj.append(float(hs[w2t[i][-1]] @ d_r))
                            yy.append(y)
                cap.edit = cap.post = None
                proj, yy = np.array(proj), np.array(yy)
                if len(set(yy)) < 2:
                    return float("nan"), float("nan")
                return roc_auc_score(yy, proj), proj[yy == 1].mean() - proj[yy == 0].mean()

            auc0, gap0 = readout_auc([], "clean")
            abl = {"criterion": crit, "clean_auc": round(auc0, 3), "clean_gap": round(gap0, 3),
                   "runs": []}
            pool = [(l, h) for l in range(r) for h in range(H)]
            for k in a.k:
                top = order[:k]
                for mode in ("direction", "zero"):
                    auc, gap = readout_auc(top, mode)
                    rr = [readout_auc([pool[i] for i in rng.choice(len(pool), k, replace=False)], mode)
                          for _ in range(a.random_draws)]
                    run = {"k": k, "mode": mode, "heads": [list(x) for x in top],
                           "auc": round(auc, 3), "retained": round(gap / gap0, 3),
                           "random_auc": [round(x[0], 3) for x in rr],
                           "random_retained": [round(x[1] / gap0, 3) for x in rr]}
                    abl["runs"].append(run)
                    print(f"  k={k:2d} {mode:9s} retained {run['retained']:.3f} "
                          f"(random {np.mean(run['random_retained']):.3f}), auc {auc:.3f}")
            ent["ablation"] = abl
            print(f"  [{crit}] top heads {order[:5]}, {time.time() - t0:.0f}s")

    np.savez(out / "heads.npz", **arrays)
    dest = f"specimens/bible-pair_{tag}_heads.json"
    write_json(dest, summary)
    print(f"-> {out / 'heads.npz'}, {dest}")


if __name__ == "__main__":
    main()
