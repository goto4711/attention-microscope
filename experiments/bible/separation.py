"""Does the feature survive when a word comes between marker and verb?

Steps 4-5 found that Jamaican Creole past and progressive reach the verb through the
model's generic previous-word heads, while English has a dedicated progressive head
(4.5, -ing -> be). If the Creole route depends on adjacency, it should weaken when an
adverb separates marker and verb; the English one should not.

Items (all verses of the aligned Bible, labelled as in step 2):
  jnt past         overt past marker directly before the verb   (im did go)
  jnt progressive  a directly before the verb                   (im a go)
  web progressive  form of be directly before the -ing verb     (he was going)

Conditions, per item (the same adverb in each language):
  original    as in the text
  separated   adverb between marker and verb  (im did jos go / he was just going)
  control     adverb just before the marker   (im jos did go / he just was going):
              the same extra word, but marker and verb stay adjacent

Measures at the verb, per condition:
  readout     projection on the variety's own probe direction (step 3) at layers 8 and 12,
              as the share of the original gap between positive and negative words that is
              left (negatives: unchanged sample words); for the JNT also on the WEB direction
  detected    share of items still on the positive side of the midpoint between the
              original positive and negative means
  attention   attention from the verb to its marker, summed over the 5 heads that attend
              most to the marker in the original condition; and the share of the verb's
              attention that goes to the adverb in the separated condition

Adverbs: jos/just, stil/still, aalwiez/always (attested between marker and verb in the
JNT). Results are reported per adverb and pooled.

  python experiments/bible/separation.py [--max-items 400]

Writes specimens/bible-pair_<model>_separation.json (numbers only).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.abspath("data/hf_cache"))

import numpy as np  # noqa: E402
import spacy  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from common import DERIVED, OUT, read_jsonl, words_with_spans, write_json  # noqa: E402
from extract import pick_device  # noqa: E402
from features import jnt_labels, web_labels  # noqa: E402
from heads import BE_FORMS, run_verse  # noqa: E402

ADVERBS = [("jos", "just"), ("stil", "still"), ("aalwiez", "always")]
JNT_PAST = {"did", "en", "ben", "neehn"}
LAYERS = [8, 12]
TOP_HEADS = 5


def items(max_items, seed):
    """(variety, feature, text, verb index, marker index) for adjacent marker + verb."""
    pairs = read_jsonl(DERIVED / "bible_pairs.jsonl")
    rng = random.Random(seed)
    rng.shuffle(pairs)
    nlp = spacy.load("en_core_web_sm")
    out = {("jnt", "past"): [], ("jnt", "progressive"): [], ("web", "progressive"): []}
    for p in pairs:
        if all(len(v) >= max_items for v in out.values()):
            break
        ws = words_with_spans(p["jnt"])
        words = [w for w, _, _ in ws]
        lab = jnt_labels(words)
        for f in ("past", "progressive"):
            for i, y in lab[f].items():
                m = i - 1
                if y != 1 or m < 0 or len(out[("jnt", f)]) >= max_items:
                    continue
                ok = words[m].lower() in JNT_PAST if f == "past" else words[m].lower() == "a"
                if ok:
                    out[("jnt", f)].append((p["jnt"], i, m))
        if len(out[("web", "progressive")]) < max_items and any(f" {b} " in f" {p['web']} " for b in BE_FORMS):
            ws = words_with_spans(p["web"])
            lab = web_labels(nlp, p["web"], ws)
            for i, y in lab["progressive"].items():
                if y == 1 and i >= 1 and ws[i - 1][0].lower() in BE_FORMS:
                    out[("web", "progressive")].append((p["web"], i, i - 1))
    return out


def insert(text, at_word, word):
    """Insert `word` before word number at_word; return new text and spans."""
    ws = words_with_spans(text)
    s = ws[at_word][1]
    new = text[:s] + word + " " + text[s:]
    return new, [(a, b) for _, a, b in words_with_spans(new)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--max-items", type=int, default=400)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tag = a.model.split("/")[-1]
    P = dict(np.load(OUT / tag / "probes.npz"))
    device = pick_device(a.device)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, attn_implementation="eager",
                                                 dtype=torch.float32).eval().to(device)
    bos = tok.eos_token_id
    t0 = time.time()
    its = items(a.max_items, a.seed)
    print({f"{v}_{f}": len(x) for (v, f), x in its.items()}, f"items ({time.time() - t0:.0f}s)", flush=True)

    # negatives: unchanged sample words, projected the same way
    neg = {}
    for v, f in its:
        rows = read_jsonl(DERIVED / f"sample_{v}.jsonl")[:400]
        vals = []
        for row in rows:
            idx = [int(i) for i, y in row["labels"][f].items() if y == 0]
            if not idx:
                continue
            o, w2t = run_verse(model, tok, bos, row["text"], row["spans"], device)
            for i in idx:
                if w2t[i]:
                    vals.append([o.hidden_states[l][0, w2t[i][-1]].cpu().numpy() for l in LAYERS])
        neg[(v, f)] = np.array(vals, dtype=np.float32)  # [N, nl, d]

    def run(text, spans, verb, marker, adv=None):
        o, w2t = run_verse(model, tok, bos, text, spans, device, attn=True)
        if not (w2t[verb] and w2t[marker]):
            return None
        t = w2t[verb][-1]
        A = torch.stack(o.attentions)[:, 0, :, t].float().cpu().numpy()  # [L, H, T]
        return {"h": np.stack([o.hidden_states[l][0, t].cpu().numpy() for l in LAYERS]),
                "to_marker": A[:, :, w2t[marker]].sum(-1),
                "to_adverb": A[:, :, w2t[adv]].sum(-1) if adv is not None and w2t[adv] else None}

    res = {"model": a.model, "layers": LAYERS, "adverbs": ADVERBS, "top_heads": TOP_HEADS, "sets": {}}
    for (v, f), lst in its.items():
        dirs = {"own": P[f"{f}_{v}"]}
        if v == "jnt":
            dirs["web"] = P[f"{f}_web"]
        recs = {"original": [], "separated": {}, "control": {}}
        for text, verb, marker in lst:
            base = run(text, [(s, e) for _, s, e in words_with_spans(text)], verb, marker)
            if base is None:
                continue
            row = {"original": base}
            for adj, adw in ADVERBS:
                adv = adj if v == "jnt" else adw
                # separated: adverb before the verb; the verb moves one word on
                t2, s2 = insert(text, verb, adv)
                row[("separated", adj)] = run(t2, s2, verb + 1, marker, adv=verb)
                # control: adverb before the marker; both move one word on
                t3, s3 = insert(text, marker, adv)
                row[("control", adj)] = run(t3, s3, verb + 1, marker + 1)
            if all(x is not None for x in row.values()):
                recs["original"].append(row["original"])
                for (cond, adj), r in ((k, x) for k, x in row.items() if k != "original"):
                    recs[cond].setdefault(adj, []).append(r)
        n = len(recs["original"])
        L, H = recs["original"][0]["to_marker"].shape
        att0 = np.mean([r["to_marker"] for r in recs["original"]], 0)
        top = [tuple(divmod(int(x), H)) for x in np.argsort(-att0.ravel())[:TOP_HEADS]]
        tm = lambda rs: float(np.mean([sum(r["to_marker"][l, h] for l, h in top) for r in rs]))
        ent = {"n": n, "n_neg": int(len(neg[(v, f)])), "top_marker_heads": [list(x) for x in top],
               "attn_to_marker_top": {"original": round(tm(recs["original"]), 3)}, "readout": {}}
        for dname, D in dirs.items():
            out = {}
            for li, l in enumerate(LAYERS):
                d = D[l]
                pos0 = np.array([r["h"][li] @ d for r in recs["original"]])
                ng = neg[(v, f)][:, li] @ d
                mid, gap0 = (pos0.mean() + ng.mean()) / 2, pos0.mean() - ng.mean()
                sign = 1 if gap0 > 0 else -1
                e = {"gap_original": round(float(gap0), 3),
                     "detected_original": round(float(np.mean(sign * (pos0 - mid) > 0)), 3)}
                for cond in ("separated", "control"):
                    pooled = []
                    for adj, rs in recs[cond].items():
                        pc = np.array([r["h"][li] @ d for r in rs])
                        pooled.append(pc)
                        e.setdefault(cond, {})[adj] = {
                            "retained": round(float((pc.mean() - ng.mean()) / gap0), 3),
                            "detected": round(float(np.mean(sign * (pc - mid) > 0)), 3)}
                    pc = np.concatenate(pooled)
                    e[cond]["pooled"] = {"retained": round(float((pc.mean() - ng.mean()) / gap0), 3),
                                         "detected": round(float(np.mean(sign * (pc - mid) > 0)), 3)}
                out[f"L{l}"] = e
            ent["readout"][dname] = out
        for cond in ("separated", "control"):
            ent["attn_to_marker_top"][cond] = round(tm([r for rs in recs[cond].values() for r in rs]), 3)
        sep = [r for rs in recs["separated"].values() for r in rs]
        ta = np.mean([r["to_adverb"] for r in sep], 0)
        ent["attn_to_adverb_top"] = round(float(sum(ta[l, h] for l, h in top)), 3)
        res["sets"][f"{v}_{f}"] = ent
        r = ent["readout"]["own"]
        print(f"{v} {f} (n {n}): own direction retained L8 sep {r['L8']['separated']['pooled']['retained']} "
              f"ctrl {r['L8']['control']['pooled']['retained']} | L12 sep {r['L12']['separated']['pooled']['retained']} "
              f"ctrl {r['L12']['control']['pooled']['retained']} | detected L12 orig {r['L12']['detected_original']} "
              f"sep {r['L12']['separated']['pooled']['detected']} | marker attn top{TOP_HEADS} {ent['attn_to_marker_top']} "
              f"to adverb {ent['attn_to_adverb_top']} | {time.time() - t0:.0f}s", flush=True)
        write_json(f"specimens/bible-pair_{tag}_separation.json", res)
    print(f"-> specimens/bible-pair_{tag}_separation.json")


if __name__ == "__main__":
    main()
