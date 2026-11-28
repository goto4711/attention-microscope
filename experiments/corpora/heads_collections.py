"""Step 4 across the collections: do the heads that serve Standard English serve the varieties?

For each collection (sample_<key>.jsonl; spaCy labels, so standard-marked forms), with the
WEB probe directions and the step-4 heads:

  generic        per head, attention from each word to the word just before it (any word):
                 the previous-word habit, and the rank of each head in this collection
  be->ing        per head, attention from a progressive -ing verb to its form of be
                 (was/were/is ...). Head 4.5 is the WEB's dedicated progressive head
  ablation       at the readout layer (12), the gap between positive and negative words along
                 the WEB direction for plural, past and progressive, and the share that
                 survives when zeroing (a) the top WEB marker heads for that feature from
                 step 4, (b) the top generic previous-word heads of this collection,
                 (c) random heads from the same layers

The WEB and JNT are included as reference points. Runs on the GPU (MPS/CUDA) when available.

  python experiments/corpora/heads_collections.py [--n 400] [--only coraal wpa]

Writes specimens/corpora_heads_<model>.json (numbers only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.abspath("data/hf_cache"))

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bible"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from common import DERIVED, FEATURES, OUT, read_jsonl, write_json  # noqa: E402
from extract import pick_device  # noqa: E402
from heads import Capture, marker_index, run_verse  # noqa: E402
from tokenisation_profile import DISTANCE, LABEL  # noqa: E402

ORDER = ["web", "ice-ja-written", "aapb-howard", "aapb-eotp", "ice-in-written", "ice-in-spoken",
         "ice-ja-spoken", "wpa", "coraal", "lince", "jnt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n", type=int, default=400, help="sentences per collection")
    ap.add_argument("--k", type=int, nargs="+", default=[5, 20])
    ap.add_argument("--random-draws", type=int, default=3)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    tag = a.model.split("/")[-1]
    out = OUT / tag
    meta = json.loads((out / "meta.json").read_text())
    L, H, hd = meta["n_layers"], meta["n_heads"], meta["d_head"]
    P = dict(np.load(out / "probes.npz"))
    step4 = json.loads(Path(f"specimens/bible-pair_{tag}_heads.json").read_text())
    r = step4["readout_layer"]
    device = pick_device(a.device)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, attn_implementation="eager",
                                                 dtype=torch.float32).eval().to(device)
    cap = Capture(model, H, hd)
    bos = tok.eos_token_id
    rng = np.random.default_rng(0)
    pool = [(l, h) for l in range(r) for h in range(H)]
    print(f"{a.model} on {device}")

    # WEB marker heads per feature (step 4 order; for plural/past WEB has no marker,
    # step 4 ranked by head decoding)
    web_heads = {f: [tuple(x) for x in max((run for run in step4["features"][f]["web"]
                                             .get("ablation", {}).get("runs", [])),
                                            key=lambda run: run["k"], default={"heads": []})["heads"]]
                 for f in FEATURES}

    dirs = {f: torch.tensor(P[f"{f}_web"][r], dtype=torch.float32, device=device) for f in FEATURES}
    res = {"model": a.model, "readout_layer": r, "n_sentences": a.n, "collections": {}}
    keys = a.only or ORDER
    for key in keys:
        sp = DERIVED / f"sample_{key}.jsonl"
        if not sp.exists():
            continue
        t0 = time.time()
        rows = read_jsonl(sp)[:a.n]
        gen, gen_n = np.zeros((L, H)), 0
        bi, bi_n = np.zeros((L, H)), 0
        for row in rows:
            o, w2t = run_verse(model, tok, bos, row["text"], row["spans"], device, attn=True)
            A = torch.stack(o.attentions)[:, 0]  # [L, H, T, T]
            for i in range(1, len(w2t)):
                if w2t[i] and w2t[i - 1]:
                    gen += A[:, :, w2t[i][-1], w2t[i - 1]].sum(-1).cpu().numpy()
                    gen_n += 1
            vocab = "jnt" if key == "jnt" else "web"
            for i, y in row["labels"]["progressive"].items():
                i = int(i)
                if y != 1 or not w2t[i]:
                    continue
                m = marker_index(vocab, "progressive", row["words"], i)
                if m is not None and w2t[m]:
                    bi += A[:, :, w2t[i][-1], w2t[m]].sum(-1).cpu().numpy()
                    bi_n += 1
        gen /= max(gen_n, 1)
        rank = np.zeros((L, H), int)
        rank.ravel()[np.argsort(-gen.ravel())] = np.arange(1, L * H + 1)
        prev_order = [tuple(divmod(int(x), H)) for x in np.argsort(-gen[:r].ravel())]
        ent = {"label": LABEL.get(key, {"jnt": "Jamaican New Testament"}.get(key, key)),
               "distance": DISTANCE.get(key), "sentences": len(rows),
               "prev_word": np.round(gen, 3).tolist(),
               "top_prev_heads": [[l, h, round(float(gen[l, h]), 3)] for l, h in prev_order[:5]],
               "head_4_5": {"prev_word": round(float(gen[4, 5]), 3), "prev_rank": int(rank[4, 5])}}
        if bi_n:
            bi /= bi_n
            ent["be_ing"] = {"n": bi_n, "attn": np.round(bi, 3).tolist(),
                             "head_4_5": round(float(bi[4, 5]), 3),
                             "top": [[int(x // H), int(x % H), round(float(bi.ravel()[x]), 3)]
                                     for x in np.argsort(-bi[:r].ravel())[:5]]}

        def gap(f, heads):
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
            pos, neg = [], []
            for row in rows:
                lab = row["labels"][f]
                if not lab:
                    continue
                o, w2t = run_verse(model, tok, bos, row["text"], row["spans"], device)
                hs = o.hidden_states[r][0]
                for i, y in lab.items():
                    if w2t[int(i)]:
                        (pos if y == 1 else neg).append(float(hs[w2t[int(i)][-1]] @ dirs[f]))
            cap.edit = None
            if len(pos) < 10 or len(neg) < 10:
                return None, len(pos), len(neg)
            return float(np.mean(pos) - np.mean(neg)), len(pos), len(neg)

        ent["ablation"] = {}
        for f in FEATURES:
            g0, npos, nneg = gap(f, [])
            if g0 is None:
                ent["ablation"][f] = {"skipped": f"too few labels ({npos}/{nneg})"}
                continue
            fe = {"n_pos": npos, "n_neg": nneg, "runs": []}
            for k in a.k:
                rand = [gap(f, [pool[i] for i in rng.choice(len(pool), k, replace=False)])[0] / g0
                        for _ in range(a.random_draws)]
                fe["runs"].append({
                    "k": k,
                    "web_marker_heads": round(gap(f, web_heads[f][:k])[0] / g0, 3) if web_heads[f] else None,
                    "prev_word_heads": round(gap(f, prev_order[:k])[0] / g0, 3),
                    "random": [round(x, 3) for x in rand]})
            ent["ablation"][f] = fe
        res["collections"][key] = ent
        b = ent.get("be_ing", {})
        print(f"{ent['label']:34s} head 4.5: be->ing {b.get('head_4_5', '-')} (n {b.get('n', 0)}), "
              f"prev-word {ent['head_4_5']['prev_word']} | " +
              " | ".join(f"{f[:4]} k20 marker {fe['runs'][-1]['web_marker_heads']} prev "
                         f"{fe['runs'][-1]['prev_word_heads']} rand {np.mean(fe['runs'][-1]['random']):.2f}"
                         for f, fe in ent["ablation"].items() if "runs" in fe) +
              f" | {time.time() - t0:.0f}s", flush=True)
        write_json(f"specimens/corpora_heads_{tag}.json", res)
    print(f"-> specimens/corpora_heads_{tag}.json")


if __name__ == "__main__":
    main()
