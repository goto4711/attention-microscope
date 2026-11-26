# Attention microscope

An interactive toy attention head. A tiny language with eight readable features (noun, verb, adjective, determiner, animate, plural, and a two-dimensional position code) goes through one head, and you can set or train **W_Q**, **W_K** and **W_V** to see which features each lens lets through and which it leaves invisible.

**Live page:** https://goto4711.github.io/attention-microscope/

Styled in the Deep Culture design; assets in `brand/` come from [goto4711/deep-culture-brand](https://github.com/goto4711/deep-culture-brand).

## What you can do

- Tap a word to see where it attends, what features arrive from the words it attends to, and what survives W_V.
- Edit the matrices cell by cell, or hide a feature by zeroing its row.
- Train W_Q and W_K in the browser (Adam, cross-entropy) towards a rule: verb → subject, adjective → noun, previous word, noun → determiner, all → animate nouns, or a pattern you paint yourself.
- Train W_V to carry chosen features, and watch W_V W_Vᵀ blur when you ask for more features than the head has dimensions.
- Inspect W_Q W_Kᵀ and W_V W_Vᵀ as feature-by-feature maps.

## Simplifications

- Embeddings are hand-set features plus a rotary-style position (angle = position × 1 rad).
- W_O is tied to W_Vᵀ so output reads in the same feature terms.
- 60 sentences are generated from a 25-word vocabulary.

Single self-contained `index.html`, no build step.
