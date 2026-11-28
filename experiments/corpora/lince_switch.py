"""Code-switching in LinCE (Hindi–English social media): what does Pythia see of the switch?

Uses the corpus's own word-level language tags (lang1 = English, lang2 = Hindi, romanised),
so no annotation is needed. Words tagged other / named entity / mixed are kept in the text
but not used as examples. A switch is a word whose language (en/hi) differs from the last
preceding en/hi word.

  language     probe for the word's language (English vs Hindi) per layer, 5-fold CV by
               sentence, next to a word-type baseline (language is largely lexical)
  switch       probe for "this word switches language" per layer, next to baselines from
               the word's type and from (previous word, word): the switch is only visible in
               context, so a probe that beats the type baseline reads context
  anticipation probe at word i for "word i+1 switches": Pythia cannot see word i+1, so
               anything above chance is the model's expectation of a switch
  heads        per head, attention to same-language words beyond what their share of the
               context predicts (+ = the head keeps to the current language, - = it reaches
               across), and attention from a switch word to the word just before it
               (across the switch) against non-switch words (previous-word habit)
  tokenisation tokens per word for English and Hindi words

Runs on the GPU (MPS/CUDA) when available.

  python experiments/corpora/lince_switch.py [--n 1500]

Writes specimens/lince_switch_<model>.json (numbers only).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.abspath("data/hf_cache"))

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bible"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from common import write_json  # noqa: E402
from extract import pick_device, word_token_map  # noqa: E402
import tokenisation_profile as tp  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
LANG = {"lang1": 0, "lang2": 1}  # 0 = English, 1 = Hindi


def sentences(n, seed):
    out = []
    for ws, ls in tp.lince():
        if not ls or any(l is None for l in ls) or not (4 <= len(ws) <= 40):
            continue
        langs = [str(l) for l in ls]
        if {"lang1", "lang2"} <= set(langs):  # keep sentences that actually mix
            out.append((ws, langs))
    random.Random(seed).shuffle(out)
    return out[:n]


def cv(X, y, g, C=0.05):
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(X, y, g):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    return round(float(balanced_accuracy_score(y, pred)), 3)


def cv_lookup(keys, y, g):
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(keys, y, g):
        by = defaultdict(Counter)
        for k, t in zip(keys[tr], y[tr]):
            by[k][t] += 1
        rate = {c: max(1, (y[tr] == c).sum()) for c in (0, 1)}
        base = Counter(y[tr]).most_common(1)[0][0]
        for i in te:
            c = by.get(keys[i])
            pred[i] = base if not c else max((0, 1), key=lambda k: c[k] / rate[k])
    return round(float(balanced_accuracy_score(y, pred)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n", type=int, default=1500, help="code-mixed sentences")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tag = a.model.split("/")[-1]
    device = pick_device(a.device)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, attn_implementation="eager",
                                                 dtype=torch.float32).eval().to(device)
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    bos = tok.eos_token_id
    sents = sentences(a.n, a.seed)
    print(f"{a.model} on {device}: {len(sents)} code-mixed sentences", flush=True)

    X, lang, sw, nxt, grp, wtype, ptype, ntok = [], [], [], [], [], [], [], []
    same_att, same_share, same_n = np.zeros((L, H)), 0.0, 0
    prev_sw, prev_sw_n, prev_ns, prev_ns_n = np.zeros((L, H)), 0, np.zeros((L, H)), 0
    t0 = time.time()
    for s, (ws, ls) in enumerate(sents):
        text, spans, pos = "", [], 0
        for w in ws:
            if text:
                text += " "
            spans.append((len(text), len(text) + len(w)))
            text += w
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = [bos] + enc["input_ids"]
        w2t = word_token_map([(0, 0)] + [tuple(o) for o in enc["offset_mapping"]], spans)
        with torch.no_grad():
            o = model(torch.tensor([ids], device=device), output_hidden_states=True, output_attentions=True)
        hs = torch.stack([o.hidden_states[l][0] for l in a.layers]).half().cpu().numpy()  # [nl, T, d]
        A = torch.stack(o.attentions)[:, 0].float().cpu().numpy()  # [L, H, T, T]
        tag_of = [LANG.get(l) for l in ls]
        last = None  # language of the last en/hi word
        lang_hist = []
        for i, w in enumerate(ws):
            li = tag_of[i]
            if not w2t[i]:
                lang_hist.append(li)
                continue
            t = w2t[i][-1]
            if li is not None:
                is_sw = int(last is not None and li != last)
                # next en/hi word
                nl = next((tag_of[j] for j in range(i + 1, len(ws)) if tag_of[j] is not None), None)
                X.append(hs[:, t])
                lang.append(li)
                sw.append(is_sw if last is not None else -1)
                nxt.append(int(nl != li) if nl is not None else -1)
                grp.append(s)
                wtype.append(w.lower())
                ptype.append(ws[i - 1].lower() if i else "<s>")
                ntok.append(len(w2t[i]))
                # heads: attention to same-language context words beyond their share
                ctx = [j for j in range(i) if tag_of[j] is not None and w2t[j]]
                if ctx:
                    same = [j for j in ctx if tag_of[j] == li]
                    tot = sum(A[:, :, t, w2t[j]].sum(-1) for j in ctx)
                    to_same = sum((A[:, :, t, w2t[j]].sum(-1) for j in same), np.zeros((L, H)))
                    same_att += to_same / np.maximum(tot, 1e-6)
                    same_share += len(same) / len(ctx)
                    same_n += 1
                if i and w2t[i - 1] and tag_of[i - 1] is not None:
                    v = A[:, :, t, w2t[i - 1]].sum(-1)
                    if tag_of[i - 1] != li:
                        prev_sw += v
                        prev_sw_n += 1
                    else:
                        prev_ns += v
                        prev_ns_n += 1
                last = li
            lang_hist.append(li)
        if (s + 1) % 250 == 0:
            print(f"  {s + 1}/{len(sents)} sentences, {time.time() - t0:.0f}s", flush=True)

    X = np.stack(X).astype(np.float32)  # [N, nl, d]
    lang, sw, nxt, grp = map(np.array, (lang, sw, nxt, grp))
    wtype, ptype, ntok = np.array(wtype), np.array(ptype), np.array(ntok)
    res = {"model": a.model, "sentences": len(sents), "words": int(len(lang)),
           "layers": a.layers,
           "tokenisation": {"english_tokens_per_word": round(float(ntok[lang == 0].mean()), 3),
                            "hindi_tokens_per_word": round(float(ntok[lang == 1].mean()), 3)}}
    print(f"words: {len(lang)} ({(lang == 0).sum()} English, {(lang == 1).sum()} Hindi); "
          f"switches {int((sw == 1).sum())}", flush=True)

    def probe(name, y, mask):
        Xm, ym, gm = X[mask], y[mask], grp[mask]
        acc = [cv(Xm[:, k], ym, gm) for k in range(len(a.layers))]
        base = {"word_type": cv_lookup(wtype[mask], ym, gm),
                "prev_and_word": cv_lookup(np.array([p + "|" + w for p, w in zip(ptype[mask], wtype[mask])]), ym, gm)}
        res[name] = {"n_pos": int(ym.sum()), "n_neg": int(len(ym) - ym.sum()), "acc": acc, "baselines": base}
        best = int(np.argmax(acc))
        print(f"{name:13s} n {int(ym.sum())}/{int(len(ym) - ym.sum())} | acc L0 {acc[0]:.3f}, best {acc[best]:.3f} "
              f"(L{a.layers[best]}) | type baseline {base['word_type']}, prev+word {base['prev_and_word']}", flush=True)

    probe("language", lang, np.ones(len(lang), bool))
    probe("switch", sw, sw >= 0)
    probe("anticipation", nxt, nxt >= 0)

    same_pref = same_att / max(same_n, 1) - same_share / max(same_n, 1)
    res["heads"] = {
        "same_language_preference": np.round(same_pref, 3).tolist(),
        "expected_same_share": round(same_share / max(same_n, 1), 3),
        "top_same_language": [[int(x // H), int(x % H), round(float(same_pref.ravel()[x]), 3)]
                              for x in np.argsort(-same_pref.ravel())[:8]],
        "top_cross_language": [[int(x // H), int(x % H), round(float(same_pref.ravel()[x]), 3)]
                               for x in np.argsort(same_pref.ravel())[:8]],
        "prev_word_at_switch": np.round(prev_sw / max(prev_sw_n, 1), 3).tolist(),
        "prev_word_no_switch": np.round(prev_ns / max(prev_ns_n, 1), 3).tolist()}
    ps, pn = prev_sw / max(prev_sw_n, 1), prev_ns / max(prev_ns_n, 1)
    top_prev = [tuple(divmod(int(x), H)) for x in np.argsort(-pn.ravel())[:5]]
    res["heads"]["top_prev_heads"] = [[l, h, round(float(pn[l, h]), 3), round(float(ps[l, h]), 3)] for l, h in top_prev]
    print("heads keeping to the current language:", res["heads"]["top_same_language"][:5])
    print("heads reaching across languages:     ", res["heads"]["top_cross_language"][:5])
    print("previous-word heads, attention to previous word without / at a switch:",
          [(l, h, a_, b_) for l, h, a_, b_ in res["heads"]["top_prev_heads"]])
    print("tokens per word:", res["tokenisation"])
    write_json(f"specimens/lince_switch_{tag}.json", res)
    print(f"-> specimens/lince_switch_{tag}.json")


if __name__ == "__main__":
    main()
