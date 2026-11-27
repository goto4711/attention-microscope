"""Annotation workbooks: export the CSV sheets to Excel for annotators, import labels back.

  python experiments/bible/annotation_xlsx.py export [--web-n 100] [--annotator NAME]
      data/annotation/*.csv -> data/annotation/annotate_jnt.xlsx, annotate_web.xlsx
  python experiments/bible/annotation_xlsx.py import data/annotation/annotate_jnt.xlsx ...
      writes yes/no/skip back into the `gold` column of the matching CSV (1/0/x)

The workbooks contain Jamaican NT text (© Bible Society of the West Indies): share them
only with collaborators for annotation, never publicly. They live in data/ (gitignored).

Annotators do not see the automatic label: a blind annotation lets us measure how
accurate the rules (JNT) and spaCy (WEB) are, not only correct them.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

from common import DERIVED, FEATURES, read_jsonl, words_with_spans

ANN = Path("data/annotation")
NT_ORDER = ["MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "1CO", "2CO", "GAL", "EPH", "PHP",
            "COL", "1TH", "2TH", "1TI", "2TI", "TIT", "PHM", "HEB", "JAS", "1PE", "2PE",
            "1JN", "2JN", "3JN", "JUD", "REV"]
TO_GOLD = {"yes": "1", "no": "0", "skip": "x"}
FONT = "Arial"
PURPLE = "832161"
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")
WORD_FILL = PatternFill("solid", fgColor="F3E3EE")
HEAD_FILL = PatternFill("solid", fgColor=PURPLE)
THIN = Side(style="thin", color="D9D9D9")

TITLES = {"plural": "Plural", "past": "Past", "progressive": "Progressive"}

QUESTIONS = {
    ("jnt", "plural"): "Does the highlighted word END a noun phrase that refers to more than one thing?",
    ("jnt", "past"): "Is the highlighted word a verb (or predicate) describing a PAST event or state?",
    ("jnt", "progressive"): "Is the highlighted word a verb describing an action IN PROGRESS (ongoing at that time)?",
    ("web", "plural"): "Is the highlighted word a noun that refers to more than one thing?",
    ("web", "past"): "Is the highlighted word a verb in the PAST tense?",
    ("web", "progressive"): "Is the highlighted word a verb in the PROGRESSIVE (be + -ing)?",
}

GUIDE = {
    "jnt": {
        "plural": [
            "yes: the noun phrase is plural. If plural 'dem' follows a noun, the highlighted 'dem' gets yes "
            "(di waiz man [dem]); a bare plural noun with no 'dem' also gets yes (aal di ed [priis]).",
            "no: the noun phrase is singular (wan [man], dis [biebi]).",
            "skip: not the last word of a noun phrase — the noun before a plural 'dem' (the 'dem' carries "
            "the label), pronoun 'dem' (tel [dem], [dem] go), verbs, adjectives, names.",
        ],
        "past": [
            "yes: past time, whether marked (did / en / ben / neehn + verb) or unmarked "
            "(Juozif [wiek] op = Joseph woke up). Use the English verse to check the time.",
            "no: present, future (wi / a-go), habitual, commands, wishes.",
            "skip: not a verb or predicate (nouns, names, function words).",
        ],
        "progressive": [
            "yes: ongoing action, usually after 'a' (im [a] go → the verb after 'a'; dem did a [tingk]).",
            "no: a verb that is not progressive (completed, habitual, future, stative).",
            "skip: not a verb. Note: 'a' is also the copula (dat a Krais) and 'to/at' (go a Jeruusilem).",
        ],
    },
    "web": {
        "plural": ["yes: plural noun (men, disciples).", "no: singular noun (man, disciple).",
                   "skip: not a noun, or a proper name."],
        "past": ["yes: past tense verb (went, said, believed).",
                 "no: present tense or base form after a modal (says, believe, will go).",
                 "skip: not a finite verb (participles, infinitives, nouns)."],
        "progressive": ["yes: -ing verb with a form of 'be' (was teaching, are going).",
                        "no: any other finite verb.",
                        "skip: not a verb, or -ing without 'be' (teaching them, he went)."],
    },
}

EXAMPLES = {
    "jnt": [("Plural", "Di Waiz Man", "dem", "", "yes"),
            ("Plural", "Di Waiz", "Man", "dem", "skip"),
            ("Past", "Den wen Juozif", "wiek", "op, im du jos wa ...", "yes"),
            ("Progressive", "Az dem", "a", "go bout dem bizniz", "skip  (the label goes on 'go')")],
    "web": [("Plural", "Then the wise", "men", "came", "yes"),
            ("Past", "When Joseph", "woke", "up", "yes")],
}


def ref_key(ref):
    book, cv = ref.split()
    c, v = cv.split(":")
    return NT_ORDER.index(book), int(c), int(v)


def style_header(ws, row, ncol):
    for c in range(1, ncol + 1):
        cell = ws.cell(row, c)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF")
        cell.fill = HEAD_FILL
        cell.alignment = Alignment(vertical="center", wrap_text=True)


def instructions(wb, variety, counts, annotator):
    ws = wb.active
    ws.title = "Instructions"
    name = "Jamaican New Testament" if variety == "jnt" else "World English Bible (Standard English check)"
    ws["A1"] = f"Annotation: {name}"
    ws["A1"].font = Font(name=FONT, bold=True, size=16, color=PURPLE)
    ws["A3"], ws["B3"] = "Annotator", annotator or ""
    ws["A4"], ws["B4"] = "Date started", ""
    for c in ("B3", "B4"):
        ws[c].fill = INPUT_FILL
    r = 6
    lines = [
        "What to do",
        "Each tab lists words in context. The word to judge is highlighted in the WORD column.",
        "Choose yes, no or skip in the yellow LABEL column (drop-down). Leave it empty if you are unsure; "
        "add a note if something is odd.",
        "Only edit the yellow cells. Do not sort, delete or add rows, and do not edit the grey ID columns.",
        "Judge meaning, not form: the question is the same as for Standard English, only the marking differs.",
    ]
    if variety == "jnt":
        lines.append("The English column gives the World English Bible verse for orientation. "
                     "The translations are free, so do not expect word-for-word matches.")
        lines.append("Confidential: the Jamaican text is © 2012 Bible Society of the West Indies. "
                     "Please do not share or upload this file.")
    for i, t in enumerate(lines):
        ws.cell(r, 1, t).font = Font(name=FONT, bold=(i == 0), size=12 if i == 0 else 10)
        r += 1
    r += 1
    for f in FEATURES:
        ws.cell(r, 1, f"Tab '{TITLES[f]}'").font = Font(name=FONT, bold=True, size=12, color=PURPLE)
        r += 1
        ws.cell(r, 1, QUESTIONS[(variety, f)]).font = Font(name=FONT, italic=True)
        r += 1
        for g in GUIDE[variety][f]:
            ws.cell(r, 1, g).font = Font(name=FONT)
            ws.cell(r, 1).alignment = Alignment(wrap_text=True, vertical="top")
            r += 1
        r += 1
    ws.cell(r, 1, "Worked examples (not part of the data)").font = Font(name=FONT, bold=True, size=12)
    r += 1
    for c, h in enumerate(["Tab", "Before", "WORD", "After", "LABEL"], 1):
        ws.cell(r, c, h)
    style_header(ws, r, 5)
    r += 1
    for ex in EXAMPLES[variety]:
        for c, v in enumerate(ex, 1):
            cell = ws.cell(r, c, v)
            cell.font = Font(name=FONT, bold=(c == 3))
            if c == 3:
                cell.fill = WORD_FILL
        r += 1
    r += 1
    ws.cell(r, 1, "Progress").font = Font(name=FONT, bold=True, size=12)
    r += 1
    for c, h in enumerate(["Tab", "Rows", "Labelled", "Done"], 1):
        ws.cell(r, c, h)
    style_header(ws, r, 4)
    r += 1
    for f in FEATURES:
        t = TITLES[f]
        ws.cell(r, 1, t).font = Font(name=FONT)
        ws.cell(r, 2, counts[f]).font = Font(name=FONT)
        ws.cell(r, 3, f"=COUNTA('{t}'!F2:F{counts[f] + 1})").font = Font(name=FONT)
        ws.cell(r, 4, f"=IF(B{r}=0,0,C{r}/B{r})").number_format = "0%"
        ws.cell(r, 4).font = Font(name=FONT)
        r += 1
    ws.column_dimensions["A"].width = 110
    for c in "BCDE":
        ws.column_dimensions[c].width = 18


def feature_sheet(wb, variety, f, rows, pairs):
    ws = wb.create_sheet(TITLES[f])
    heads = ["Verse", "Before", "WORD", "After"] + (["English (WEB)"] if variety == "jnt" else []) \
        + ["LABEL", "Note", "ID: ref", "ID: word"]
    # keep LABEL in column F for both workbooks
    if variety == "web":
        heads = ["Verse", "Before", "WORD", "After", "Whole verse", "LABEL", "Note", "ID: ref", "ID: word"]
    for c, h in enumerate(heads, 1):
        ws.cell(1, c, h)
    style_header(ws, 1, len(heads))
    ws.cell(1, 6).comment = Comment(QUESTIONS[(variety, f)] + "  yes / no / skip", "pipeline")
    dv = DataValidation(type="list", formula1='"yes,no,skip"', allow_blank=True,
                        showErrorMessage=True, errorTitle="Label",
                        error="Choose yes, no or skip (or leave empty).")
    ws.add_data_validation(dv)
    for i, r in enumerate(rows, 2):
        text = pairs[r["ref"]][variety]
        words = [w for w, _, _ in words_with_spans(text)]
        k = int(r["word_idx"])
        before = " ".join(words[max(0, k - 12):k])
        after = " ".join(words[k + 1:k + 13])
        other = pairs[r["ref"]]["web"] if variety == "jnt" else text
        vals = [r["ref"], before, words[k], after, other, TO_LABEL.get(r.get("gold", ""), ""),
                "", r["ref"], k]
        for c, v in enumerate(vals, 1):
            cell = ws.cell(i, c, v)
            cell.font = Font(name=FONT, size=10, bold=(c == 3),
                             color="808080" if c >= 8 else "000000")
            cell.alignment = Alignment(vertical="top", wrap_text=c in (2, 4, 5, 7),
                                       horizontal="right" if c == 2 else None)
            cell.border = Border(bottom=THIN)
        ws.cell(i, 3).fill = WORD_FILL
        ws.cell(i, 6).fill = INPUT_FILL
        ws.cell(i, 7).fill = INPUT_FILL
        dv.add(ws.cell(i, 6))
    widths = [11, 42, 14, 42, 55, 9, 24, 10, 7]
    for c, w in enumerate(widths, 1):
        ws.column_dimensions[ws.cell(1, c).column_letter].width = w
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:{ws.cell(1, len(heads)).column_letter}{len(rows) + 1}"


TO_LABEL = {v: k for k, v in TO_GOLD.items()}


def read_sheet(variety, f):
    with open(ANN / f"{variety}_{f}.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def export(a):
    pairs = {p["ref"]: p for p in read_jsonl(DERIVED / "bible_pairs.jsonl")}
    rng = random.Random(a.seed)
    for variety in ("jnt", "web"):
        wb = Workbook()
        chosen = {}
        for f in FEATURES:
            rows = read_sheet(variety, f)
            if variety == "web" and a.web_n and len(rows) > a.web_n:
                rows = rng.sample(rows, a.web_n)
            chosen[f] = sorted(rows, key=lambda r: (ref_key(r["ref"]), int(r["word_idx"])))
        instructions(wb, variety, {f: len(chosen[f]) for f in FEATURES}, a.annotator)
        for f in FEATURES:
            feature_sheet(wb, variety, f, chosen[f], pairs)
        wb.calculation.fullCalcOnLoad = True  # progress counters update when opened
        out = ANN / f"annotate_{variety}.xlsx"
        wb.save(out)
        print(f"{out}: " + ", ".join(f"{TITLES[f]} {len(chosen[f])}" for f in FEATURES))


def import_(a):
    for path in a.files:
        wb = load_workbook(path)
        variety = "jnt" if "jnt" in Path(path).stem else "web"
        for f in FEATURES:
            ws = wb[TITLES[f]]
            got = {}
            for row in ws.iter_rows(min_row=2, values_only=True):
                ref, k, lab = row[7], row[8], (row[5] or "").strip().lower()
                if ref and lab:
                    if lab not in TO_GOLD:
                        raise SystemExit(f"{path} {TITLES[f]}: unknown label {lab!r} at {ref}")
                    got[(ref, int(k))] = (TO_GOLD[lab], (row[6] or "").strip())
            rows = read_sheet(variety, f)
            fields = list(rows[0].keys()) + (["note"] if "note" not in rows[0] else [])
            n = 0
            for r in rows:
                g = got.get((r["ref"], int(r["word_idx"])))
                if g:
                    r["gold"], r["note"] = g
                    n += 1
            with open(ANN / f"{variety}_{f}.csv", "w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            print(f"{path} {TITLES[f]}: {n} labels -> {variety}_{f}.csv")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--web-n", type=int, default=100, help="rows per WEB tab (0 = all)")
    e.add_argument("--annotator", default="")
    e.add_argument("--seed", type=int, default=0)
    i = sub.add_parser("import")
    i.add_argument("files", nargs="+")
    a = ap.parse_args()
    export(a) if a.cmd == "export" else import_(a)


if __name__ == "__main__":
    main()
