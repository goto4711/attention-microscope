"""Step 1: per-head weights and activations from Pythia on the Bible pair.

Writes to experiments/out/<model>/ (gitignored; it holds no corpus text, but it is large):

  weights.npz     per layer l and head h, as float16:
                    W_Q, W_K, W_V   [L, H, d_head, d_model]   (rows = head dims)
                    b_Q, b_K, b_V   [L, H, d_head]
                    W_O             [L, H, d_model, d_head]
                    ln_g, ln_b      [L, d_model]   input LayerNorm of each layer
                  W_Q/W_K/W_V act on LayerNorm(x), not on x; the visibility
                  analysis (step 4) has to fold ln_g in. Pythia rotates the first
                  `rotary_ndims` dims of each q/k head; that is recorded in meta.
  acts_<variety>.npz
                  X       [N, L+1, d_model] float16: residual stream *entering* each
                          layer (index L = final, before the last LayerNorm), at the
                          last subtoken of each labelled word only
                  verse, word   [N] indices into sample_<variety>.jsonl
                  tokens_per_word  [n_words_total] for the tokenisation control
  attn_<variety>.npz
                  for the first --attn-n verses: word-level attention [L, H, W, W]
                  (to-word = sum over its subtokens, from-word = mean over its
                  subtokens, as in Clark et al. 2019) and bos [L, H, W], the mass
                  sent to the BOS token (attention sink), as float16
  meta.json

A BOS token (<|endoftext|>) is prepended so the first word is not the attention sink.
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("HF_HOME", os.path.abspath("data/hf_cache"))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import DERIVED, FEATURES, OUT, read_jsonl, write_json


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def export_weights(model, path):
    cfg = model.config
    L, H, d = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size
    hd = d // H
    out = {k: np.zeros(s, np.float16) for k, s in {
        "W_Q": (L, H, hd, d), "W_K": (L, H, hd, d), "W_V": (L, H, hd, d),
        "b_Q": (L, H, hd), "b_K": (L, H, hd), "b_V": (L, H, hd),
        "W_O": (L, H, d, hd), "ln_g": (L, d), "ln_b": (L, d)}.items()}
    for l, layer in enumerate(model.gpt_neox.layers):
        att = layer.attention
        W = att.query_key_value.weight.detach().float().view(H, 3, hd, d).cpu().numpy()
        b = att.query_key_value.bias.detach().float().view(H, 3, hd).cpu().numpy()
        for k, name in enumerate("QKV"):
            out[f"W_{name}"][l] = W[:, k]
            out[f"b_{name}"][l] = b[:, k]
        Wo = att.dense.weight.detach().float().cpu().numpy()  # [d, H*hd]
        out["W_O"][l] = Wo.reshape(d, H, hd).transpose(1, 0, 2)
        out["ln_g"][l] = layer.input_layernorm.weight.detach().float().cpu().numpy()
        out["ln_b"][l] = layer.input_layernorm.bias.detach().float().cpu().numpy()
    np.savez(path, **out)


def word_token_map(offsets, spans):
    """For each word, the indices of tokens overlapping it (token 0 is BOS)."""
    toks = []
    for s, e in spans:
        toks.append([t for t, (a, b) in enumerate(offsets) if b > a and a < e and b > s])
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--revision", default="main", help="e.g. step143000; for training-time analyses")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--attn-n", type=int, default=60, help="verses whose attention is saved")
    ap.add_argument("--limit", type=int, default=0, help="only the first N verses (0 = all)")
    a = ap.parse_args()

    device = pick_device(a.device)
    tag = a.model.split("/")[-1] + ("" if a.revision == "main" else f"@{a.revision}")
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, revision=a.revision, attn_implementation="eager", dtype=torch.float32)
    model.eval().to(device)
    cfg = model.config
    L, H, d = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size
    print(f"{a.model}@{a.revision} on {device}: {L} layers x {H} heads, d_model={d}")

    export_weights(model, out / "weights.npz")
    bos = tok.eos_token_id  # Pythia uses <|endoftext|> as document separator

    meta = {"model": a.model, "revision": a.revision, "n_layers": L, "n_heads": H,
            "d_model": d, "d_head": d // H,
            "rotary_ndims": int(model.gpt_neox.layers[0].attention.rotary_ndims),
            "use_parallel_residual": bool(cfg.use_parallel_residual),
            "residual_index": "X[:, l] is the residual stream entering layer l; X[:, L] is the output of the last layer",
            "word_position": "last subtoken of the word",
            "varieties": {}}

    for variety in ("web", "jnt"):
        rows = read_jsonl(DERIVED / f"sample_{variety}.jsonl")
        if a.limit:
            rows = rows[:a.limit]
        X, vi, wi, tpw = [], [], [], []
        attn = {}
        t0 = time.time()
        for v, r in enumerate(rows):
            enc = tok(r["text"], return_offsets_mapping=True, add_special_tokens=False)
            ids = [bos] + enc["input_ids"]
            offsets = [(0, 0)] + [tuple(o) for o in enc["offset_mapping"]]
            w2t = word_token_map(offsets, r["spans"])
            tpw.extend(len(t) for t in w2t)
            want = sorted({int(i) for f in FEATURES for i in r["labels"][f]})
            with torch.no_grad():
                o = model(torch.tensor([ids], device=device),
                          output_hidden_states=True, output_attentions=v < a.attn_n)
            hs = torch.stack(o.hidden_states, 1)[0]  # [L+1, T, d]
            for i in want:
                if w2t[i]:
                    X.append(hs[:, w2t[i][-1]].half().cpu().numpy())
                    vi.append(v)
                    wi.append(i)
            if v < a.attn_n:
                A = torch.stack(o.attentions, 0)[:, 0].float()  # [L, H, T, T]
                W = len(w2t)
                Aw = torch.zeros(L, H, W, W, device=A.device)
                for qi, qt in enumerate(w2t):
                    if not qt:
                        continue
                    rowmean = A[:, :, qt, :].mean(2)  # [L, H, T]
                    for ki, kt in enumerate(w2t):
                        if kt:
                            Aw[:, :, qi, ki] = rowmean[:, :, kt].sum(-1)
                bos_mass = torch.stack([A[:, :, qt, 0].mean(2) if qt else
                                        torch.zeros(L, H, device=A.device) for qt in w2t], -1)
                attn[f"a{v}"] = Aw.half().cpu().numpy()
                attn[f"bos{v}"] = bos_mass.half().cpu().numpy()
            if (v + 1) % 100 == 0:
                print(f"  {variety}: {v + 1}/{len(rows)} verses, {time.time() - t0:.0f}s")
        np.savez(out / f"acts_{variety}.npz", X=np.stack(X), verse=np.array(vi),
                 word=np.array(wi), tokens_per_word=np.array(tpw))
        np.savez_compressed(out / f"attn_{variety}.npz", **attn)
        tpw = np.array(tpw)
        meta["varieties"][variety] = {"verses": len(rows), "labelled_words": len(X),
                                      "tokens_per_word_mean": float(tpw.mean()),
                                      "tokens_per_word_hist": np.bincount(tpw).tolist()}
        print(f"{variety}: {len(X)} labelled words, {tpw.mean():.2f} tokens/word, "
              f"{time.time() - t0:.0f}s")
    write_json(out / "meta.json", meta)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
