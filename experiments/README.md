# Experiments: what attention heads see in varieties of English

Do a language model's attention heads see the same grammatical features in varieties
of English as in Standard English, or do they leave some invisible, and does that
change with distance from the standard?

This folder holds the Python pipeline. The page (`index.html`) stays a self-contained
static file; the pipeline only feeds it derived numbers through `specimens/`.

## Layout

```
experiments/
  requirements.txt     pinned versions (tested: Python 3.13, macOS, CPU)
  bible/               controlled core: Jamaican NT x World English Bible, verse by verse
    run.sh             steps 1-3 end to end
    parse_jnt.py       JNT PDF text -> verses
    align.py           JNT <-> WEB by book:chapter:verse
    features.py        step 2: word-level feature labels + annotation sheets
    extract.py         step 1: per-head W_Q/W_K/W_V/W_O, residual stream, attention
    probe.py           step 3: linear probes per layer, cross-variety transfer
    annotation_xlsx.py Excel workbooks for hand annotation, and import of the labels
  out/                 (gitignored) arrays: weights, activations, attention, probe directions
data/                  (gitignored) raw corpora, derived text, annotation sheets, HF cache
specimens/             shareable derived outputs only (JSON, no restricted corpus text)
```

## Running the Bible pair

```bash
pip install -r experiments/requirements.txt
bash experiments/bible/run.sh              # 1000 verse pairs, ~10 min on a laptop CPU
bash experiments/bible/run.sh --limit 50   # quick trial
```

`run.sh` expects `data/patois/jnt.txt` and downloads the WEB (public domain, ebible.org)
and `EleutherAI/pythia-410m` (~900 MB, cached in `data/hf_cache`). Other Pythia sizes:
`MODEL=EleutherAI/pythia-1b bash experiments/bible/run.sh`. A training checkpoint:
`python experiments/bible/extract.py --revision step64000`. If the Hugging Face download
fails with a Xet error, set `HF_HUB_DISABLE_XET=1`.

## Features and labels (step 2)

| Feature | WEB marking | JNT marking | Where the label sits |
|---|---|---|---|
| plural | noun -s | postposed *dem* (*di man dem*) | the word that ends the noun phrase: WEB noun, JNT *dem* |
| past | verb inflection | preverbal *did / en / ben / neehn* | the verb |
| progressive | *be* + V-ing | preverbal *a* (*im a go*) | the verb |

Pythia is causal: a word only sees what comes before it. A JNT noun cannot see the
*dem* after it, so the plural label is placed on *dem*, the first position where the
plural is visible.

Label sources are recorded per word (`spacy`, `rule`, `gold`):

- **WEB**: spaCy `en_core_web_sm`. Its accuracy on this text still has to be measured on
  the WEB annotation sheets.
- **JNT**: rules on the overt Creole markers. They label only overt marking, so bare
  plurals and unmarked past (common in JNT narrative) stay unlabelled. Those are what the
  hand annotation adds.
- **gold**: `features.py` writes sheets to `data/annotation/<variety>_<feature>.csv`
  (balanced rule positives, rule negatives and unlabelled content words). Fill `gold`
  with `1`, `0` or `x` (not a candidate) and re-run; gold overrides the automatic label.
- **Workbooks for annotators**: `python experiments/bible/annotation_xlsx.py export` turns
  the sheets into `data/annotation/annotate_jnt.xlsx` (all JNT rows, with the WEB verse
  for orientation) and `annotate_web.xlsx` (100 rows per feature, to measure spaCy). Each
  has an Instructions tab, a yes/no/skip drop-down and a progress count. The automatic
  label is hidden so the annotation is blind. Read the labels back with
  `python experiments/bible/annotation_xlsx.py import data/annotation/annotate_jnt.xlsx`,
  then re-run `run.sh`. The JNT workbook contains copyrighted verse text: share it only
  with collaborators, never publicly.

## Outputs of step 1 and 3

- `out/<model>/weights.npz`: per head W_Q, W_K, W_V `[L, H, d_head, d_model]`, biases,
  W_O `[L, H, d_model, d_head]`, and each layer's input LayerNorm gain/bias (the heads
  read LayerNorm(x), so step 4 must fold `ln_g` into the visibility norms).
- `out/<model>/acts_<variety>.npz`: residual stream entering every layer at each labelled
  word (last subtoken), plus tokens per word for the tokenisation control.
- `out/<model>/attn_<variety>.npz`: word-level attention for the first 60 verses
  (to-word summed over subtokens, from-word averaged, as in Clark et al. 2019), plus
  mass on the BOS token.
- `out/<model>/probes.npz`: probe directions `[L+1, d_model]` per feature and variety.
  These are the real counterpart of the toy page's feature rows.
- `specimens/bible-pair_<model>_probes.json`: probe accuracy per layer, cross-variety
  transfer AUC, WEB/JNT direction cosine, label counts and sources, tokens per word.

## Caveats so far

- The JNT is a free, idiomatic translation, so verse pairs are not strict minimal pairs.
  `features.py` keeps pairs whose word-count ratio is at most 2 (6,812 of 7,933).
- Before hand annotation, WEB and JNT labels come from different procedures (a tagger
  and rules), so differences in probe accuracy partly reflect labelling.
- Word positions use the last subtoken. JNT averages ~1.5 tokens per word against ~1.1 for
  WEB; step 5 will compare first, last and mean subtoken.

## Data and licences

| Folder in `data/` | Collection | Licence (as far as known) | May go in `specimens/` |
|---|---|---|---|
| `patois/` | Jamaican New Testament, © 2012 Bible Society of the West Indies | copyright | numbers only, no verse text |
| `web/` | World English Bible (ebible.org) | public domain | yes |
| `ice-jamaica/`, `ice-india/` | International Corpus of English | ICE licence, no redistribution | numbers only |
| `storycorps/` | AAPB, Howard University Television civil-rights oral histories (WHUT) | to check | numbers only until checked |
| `aapb/eotp-interviews/` | AAPB interviews | to check | numbers only until checked |
| `coraal/` | CORAAL (DCA, ROC) | CC BY-NC-SA (to confirm) | short quotes with attribution |
| `wpa_narratives/` | WPA slave narratives (Federal Writers' Project) | public domain | yes |
| `lince_hf/` | LinCE Hindi-English | to check (platform terms) | numbers only until checked |
