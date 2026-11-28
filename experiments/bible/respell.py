"""Step 5 control: the Jamaican NT respelled in English-based spelling, grammar kept.

The JNT is written in the phonemic Cassidy–Le Page orthography (piipl, Jiizas, faada).
Pythia's tokenizer cuts these words into many more pieces than English spellings
(1.52 tokens per word against 1.08 for the WEB). To separate orthography from grammar,
each JNT word is mapped one-to-one to its English cognate spelling where there is one
(piipl -> people, faada -> father), while the grammatical markers the study is about and
Creole words without an English cognate stay as they are (dem, did, a, en, neehn, wi,
a-go, fi, im, mi, unu, se, pikni, nyam). Word positions and labels therefore carry over
unchanged: the result is a third variety, "jntr", that differs from the JNT in spelling
only and from the WEB in grammar.

Lexicon: hand-written for the frequent words (below), then an automatic cognate match
for the rest: the English candidate must occur in the aligned WEB verses and share the
consonant skeleton (after Creole sound correspondences: th -> t/d, non-rhotic r, h-drop,
z -> s). Ambiguous forms (rait = right/write) are resolved per occurrence by which
candidate the aligned WEB verse contains.

Writes data/derived/jnt_respell_lexicon.tsv (JNT word, respelling, source), and
data/derived/sample_jntr.jsonl (same verses, words, labels as sample_jnt.jsonl).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from common import DERIVED, read_jsonl, words_with_spans, write_jsonl

# Kept as written: grammatical markers and Creole words with no English cognate spelling.
KEEP = {"beks", "de-so", "fi'im", "gweehn", "baka", "aafa", "unuself", "at", "laka", "dem", "a", "fi", "im", "unu", "mi", "se", "wi", "did", "no", "wan", "de", "ya",
        "a-go", "naa", "neehn", "en", "ben", "afi", "don", "pikni", "nyam", "gwaan", "outa",
        "siek-a", "kain-a", "infronta", "sidong", "iin", "ar", "da", "unu", "dege-dege"}

HAND = {
    "di": "the", "an": "and", "we": "where", "gad": "God", "iina": "in", "piipl": "people",
    "tu": "to", "jiizas": "Jesus", "yu": "you", "du": "do", "so": "so", "mek": "make",
    "go": "go", "kom": "come", "op": "up", "wen": "when", "wa": "what", "aal": "all",
    "bot": "but", "it": "it", "kaaz": "because", "wid": "with", "dat": "that", "get": "get",
    "pan": "upon", "bout": "about", "gi": "give", "sitn": "something", "fram": "from",
    "laik": "like", "tel": "tell", "nuo": "know", "av": "have", "si": "see", "ou": "how",
    "ef": "if", "man": "man", "tek": "take", "taim": "time", "gud": "good", "kyan": "can",
    "waahn": "want", "liv": "live", "den": "then", "bak": "back", "biliiv": "believe",
    "siem": "same", "nou": "now", "ier": "hear", "out": "out", "faada": "father",
    "dong": "down", "notn": "nothing", "laad": "Lord", "taak": "talk", "krais": "Christ",
    "lov": "love", "luk": "look", "som": "some", "neva": "never", "nobadi": "nobody",
    "juu": "Jew", "tan": "stand", "ton": "turn", "dis": "this", "sumadi": "somebody",
    "sin": "sin", "shuo": "show", "spirit": "spirit", "aks": "ask", "jos": "just",
    "laa": "law", "tingk": "think", "die": "day", "evribadi": "everybody", "afta": "after",
    "kyaahn": "can't", "eva": "ever", "iivn": "even", "staat": "start", "nof": "enough",
    "paat": "part", "falara": "follower", "breda": "brother", "uova": "over", "onggl": "only",
    "evriting": "everything", "ous": "house", "fala": "follow", "uu": "who", "aat": "heart",
    "apm": "happen", "ada": "other", "wok": "work", "put": "put", "shi": "she", "big": "big",
    "wod": "word", "tiich": "teach", "wuda": "would", "powa": "power", "badi": "body",
    "kil": "kill", "tap": "stop", "worl": "world", "kaal": "call", "prie": "pray",
    "elp": "help", "bwai": "boy", "siev": "save", "riili": "really", "chos": "trust",
    "pik": "pick", "paal": "Paul", "ed": "head", "api": "happy", "uman": "woman", "dan": "than",
    "niem": "name", "rang": "wrong", "choch": "church", "evn": "even", "kaa": "because",
    "alaiv": "alive", "aneda": "another", "priich": "preach", "muo": "more", "piita": "Peter",
    "laif": "life", "sista": "sister", "kyari": "carry", "enibadi": "anybody", "gens": "against",
    "liida": "leader", "ienjel": "angel", "yuuz": "use", "chrang": "strong", "joj": "judge",
    "til": "till", "wich": "which", "mesij": "message", "sofa": "suffer", "bi": "be",
    "kuda": "could", "duo": "though", "aaf": "off", "beta": "better", "priis": "priest",
    "bad": "bad", "andastan": "understand", "kip": "keep", "uoli": "holy", "ansa": "answer",
    "fos": "first", "jan": "John", "likl": "little", "ruul": "rule", "komiin": "coming",
    "aad": "hard", "ron": "run", "ort": "hurt", "chrai": "try", "muor": "more",
    "wikid": "wicked", "wel": "well", "pramis": "promise", "bifuo": "before",
    "prafit": "prophet", "plies": "place", "main": "mind", "evri": "every", "shuor": "sure",
    "fuud": "food", "tingz": "things", "mongks": "among", "lisn": "listen",
    "jeruusilem": "Jerusalem", "sorvant": "servant", "agen": "again", "roun": "round",
    "tiicha": "teacher", "king": "king", "sopm": "something", "yai": "eye", "nyuuz": "news",
    "lait": "light", "iip": "heap", "stil": "still", "fried": "afraid", "powaful": "powerful",
    "uona": "owner", "kroud": "crowd", "kudn": "couldn't", "baal": "bawl", "beg": "beg",
    "worship": "worship", "wail": "while", "suoja": "soldier", "kraas": "cross",
    "moni": "money", "yaad": "yard", "iebriyam": "Abraham", "ting": "thing", "mada": "mother",
    "mos": "must", "priez": "praise", "kindom": "kingdom", "mii": "me", "dem-de": "those",
    "wie": "way", "bies": "base", "chruu": "through", "chuuz": "choose", "dout": "doubt",
    "ongl": "only", "ongggl": "only", "aalwiez": "always", "nevaa": "never", "agens": "against",
    "bifuor": "before", "sliip": "sleep", "chravl": "travel", "fiil": "feel",
    "fa": "for", "miin": "mean", "demself": "themselves", "jrap": "drop", "rispek": "respect",
    "brok": "break", "speshal": "special", "bada": "bother", "kech": "catch", "non": "none",
    "chuut": "truth", "waak": "walk", "jringk": "drink", "memba": "remember", "kliin": "clean",
    "bon": "burn", "kanchuol": "control", "gyada": "gather", "kain": "kind", "kluoz": "clothes",
    "mash": "mash", "toun": "town", "dizorv": "deserve", "faiya": "fire", "paadn": "pardon",
    "marid": "marry", "els": "else", "ponish": "punish", "biebi": "baby", "nyuu": "new",
    "kier": "care", "uopm": "open", "niida": "neither", "chobl": "trouble",
    "baan": "born", "farisii": "Pharisee", "sliev": "slave", "plan": "plan", "fling": "fling",
    "tugeda": "together", "wach": "watch", "set": "set", "iit": "eat", "fait": "fight",
    "fambili": "family", "aada": "order", "bilang": "belong", "yuus": "use", "uon": "own",
    "neda": "other", "mata": "matter", "bring": "bring", "bier": "bear", "mout": "mouth",
    "plaant": "plant", "wudn": "wouldn't", "sins": "since", "az": "as", "chienj": "change",
    "bitwiin": "between", "tangks": "thanks", "impuotant": "important", "sed": "said",
    "gruo": "grow", "mirikl": "miracle", "rich": "rich", "difrant": "different",
    "touzn": "thousand", "suun": "soon", "brait": "bright", "chriit": "treat", "yet": "yet",
    "chat": "chat",
}
# forms with two English cognates: resolved per verse from the WEB text
# Only base forms: JNT nouns and verbs are uninflected, so a respelling must not add English
# plural or past marking (that would leak the very features under study into the control).
AMBIG = {"rait": ["right", "write"], "uol": ["whole", "old", "hold"],
         "chuu": ["true", "through"], "lef": ["left", "leave"], "ded": ["dead", "die"],
         "chrii": ["three", "tree"], "fain": ["find", "fine"],
         "piis": ["peace", "piece"], "riich": ["reach", "rich"], "aan": ["on", "own"],
         "tuu": ["too", "two"]}

EN_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")


def skeleton_jnt(w):
    w = w.lower().replace("-", "")
    w = re.sub(r"ng(?=g)", "n", w)
    w = w.replace("z", "s").replace("h", "")
    w = re.sub(r"[aeiouy]+", "", w)
    return re.sub(r"(.)\1+", r"\1", w)


def skeleton_en(w):
    w = w.lower()
    w = w.replace("ph", "f").replace("ck", "k").replace("qu", "kw").replace("x", "ks")
    w = re.sub(r"^kn", "n", w).replace("wh", "w").replace("gh", "")
    w = w.replace("th", "d").replace("tch", "ch").replace("dge", "j")
    w = re.sub(r"c(?=[eiy])", "s", w).replace("c", "k")
    w = re.sub(r"g(?=[ei])", "j", w)
    w = re.sub(r"r(?![aeiouy])", "", w)  # non-rhotic
    w = re.sub(r"e$", "", w).replace("z", "s").replace("h", "")
    w = re.sub(r"[aeiouy]+", "", w)
    return re.sub(r"(.)\1+", r"\1", w)


def build_lexicon(pairs, min_sim=0.75, min_dice=0.25):
    jcount, ecount, co = Counter(), Counter(), defaultdict(Counter)
    for p in pairs:
        jw = {w.lower() for w, _, _ in words_with_spans(p["jnt"]) if any(c.isalpha() for c in w)}
        ew = {w.lower() for w in EN_WORD.findall(p["web"])}
        jcount.update(jw)
        ecount.update(ew)
        for j in jw:
            co[j].update(ew)
    lex = {}
    for j, nj in jcount.items():
        if j in KEEP:
            lex[j] = (j, "keep")
        elif j in HAND:
            lex[j] = (HAND[j], "hand")
        elif j in AMBIG:
            lex[j] = (AMBIG[j][0], "ambiguous")
        else:
            sj, best = skeleton_jnt(j), None
            for e, n in co[j].most_common(200):
                if not sj:
                    break
                sim = SequenceMatcher(None, sj, skeleton_en(e)).ratio()
                dice = 2 * n / (nj + ecount[e])
                if sim >= min_sim and dice >= min_dice and e[0] in "aeiouh" + j[0] or \
                        (sim >= min_sim and dice >= min_dice and skeleton_en(e)[:1] == sj[:1]):
                    score = sim + dice
                    if best is None or score > best[1]:
                        best = (e, score)
            lex[j] = (lemma(best[0]), "auto") if best else (j, "unmatched")
    return lex, jcount


_NLP = None


def lemma(e):
    """Base form of an English match (apostles -> apostle, sold -> sell, feet -> foot)."""
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    t = _NLP(e)[0]
    return t.lemma_ if t.lemma_ and t.lemma_ != "-PRON-" else e


def respell_word(w, lex, web_words):
    lw = w.lower()
    if lw in AMBIG:
        cand = next((c for c in AMBIG[lw] if c in web_words), AMBIG[lw][0])
    elif lw in lex:
        cand = lex[lw][0]
    else:
        return w
    if w[:1].isupper() and not cand[:1].isupper():
        cand = cand[:1].upper() + cand[1:]
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=0, help="print N automatic matches for checking")
    a = ap.parse_args()
    pairs = read_jsonl(DERIVED / "bible_pairs.jsonl")
    web_by_ref = {p["ref"]: {w.lower() for w in EN_WORD.findall(p["web"])} for p in pairs}
    lex, jcount = build_lexicon(pairs)

    with open(DERIVED / "jnt_respell_lexicon.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["jnt", "respelled", "source", "count"])
        for j, n in jcount.most_common():
            w.writerow([j, lex[j][0], lex[j][1], n])
    tot = sum(jcount.values())
    cov = Counter()
    for j, n in jcount.items():
        cov[lex[j][1]] += n
    print("token share by source: " + ", ".join(f"{k} {v / tot:.1%}" for k, v in cov.most_common()))
    if a.show:
        auto = [(j, lex[j][0], jcount[j]) for j in jcount if lex[j][1] == "auto"]
        for j, e, n in sorted(auto, key=lambda x: -x[2])[:a.show]:
            print(f"  {j:14s} -> {e:14s} {n}")
        un = [(j, jcount[j]) for j in jcount if lex[j][1] == "unmatched"]
        print("most frequent unmatched:", sorted(un, key=lambda x: -x[1])[:40])

    rows = read_jsonl(DERIVED / "sample_jnt.jsonl")
    out = []
    for r in rows:
        web_words = web_by_ref.get(r["ref"], set())
        parts, spans, pos = [], [], 0
        text, prev_end = "", 0
        new_words = []
        for wd, (s, e) in zip(r["words"], r["spans"]):
            nw = respell_word(wd, lex, web_words) if any(c.isalpha() for c in wd) else wd
            text += r["text"][prev_end:s]
            spans.append([len(text), len(text) + len(nw)])
            text += nw
            prev_end = e
            new_words.append(nw)
        text += r["text"][prev_end:]
        assert [w for w, _, _ in words_with_spans(text)] == new_words, (r["ref"], text)
        out.append({**r, "variety": "jntr", "text": text, "words": new_words, "spans": spans})
    write_jsonl(DERIVED / "sample_jntr.jsonl", out)
    print(f"-> {DERIVED / 'jnt_respell_lexicon.tsv'}, {DERIVED / 'sample_jntr.jsonl'} ({len(out)} verses)")
    print("example:", rows[1]["text"][:120], "\n      ->", out[1]["text"][:120])


if __name__ == "__main__":
    main()
