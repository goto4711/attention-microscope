"""Second annotator on a small subset: inter-annotator agreement with the first annotator (CA).

  python experiments/bible/agreement_xlsx.py export [--per-tab 20] [--seed 0]
      -> data/annotation/annotate_second.xlsx
  python experiments/bible/agreement_xlsx.py import data/annotation/annotate_second_<initials>.xlsx
      -> specimens/agreement_report.json (counts only) and a printed table

The subset is drawn from words the first annotator labelled, stratified by her answer so
that yes, no and skip all occur (the second annotator does not see it):
  JNT Plural / Past / Progressive   from the Jamaican NT workbook (rule-sampled words)
  Plural, no -s / Past, no -ed      pooled from the CORAAL, WPA and ICE Jamaica spoken workbooks

Import does not change any labels: it only compares. Agreement is reported per tab as
raw agreement and Cohen's kappa, over yes/no/skip and over the words both call yes or no.

The workbook contains Jamaican NT text (© Bible Society of the West Indies) and corpus
text for research use only: share it only with the annotator.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font
from openpyxl.worksheet.datavalidation import DataValidation

from annotation_xlsx import (FONT, GUIDE, INPUT_FILL, PURPLE, QUESTIONS, THIN, TO_GOLD, WORD_FILL,
                             style_header)
from common import DERIVED, read_jsonl, write_json

ANN = Path("data/annotation")
VARIETY_SETS = ["coraal", "wpa", "ice-ja-spoken"]
VARIETY_TABS = {
    "plural_unmarked": ("Plural, no -s", "Does the highlighted noun refer to MORE THAN ONE thing?",
                        ["yes: plural in meaning although it has no -s (two [mile], them [boy]).",
                         "no: it refers to one thing, or it is a mass noun (all [day], some [money]).",
                         "skip: not a noun, a fixed expression (o'clock), or you cannot tell."]),
    "past_unmarked": ("Past, no -ed", "Does the highlighted verb describe a PAST event or state?",
                      ["yes: past in meaning although the verb has no past form (he [come] up there yesterday).",
                       "no: present, future, habitual or general (you [know], they [grow] up thinking).",
                       "skip: not a verb, or you cannot tell from the sentence."]),
}
JNT_TABS = {"plural": "JNT Plural", "past": "JNT Past", "progressive": "JNT Progressive"}
LABEL_COL = 6


def split_context(ctx):
    words = ctx.split(" ")
    j = next(n for n, w in enumerate(words) if w.startswith("[") and w.endswith("]"))
    return " ".join(words[:j]), words[j][1:-1], " ".join(words[j + 1:])


def stratified(rows, n, rng, key="gold"):
    by = {}
    for r in rows:
        by.setdefault(r[key], []).append(r)
    for v in by.values():
        rng.shuffle(v)
    out, cats = [], sorted(by)
    i = 0
    while len(out) < n and any(by.values()):
        c = cats[i % len(cats)]
        if by[c]:
            out.append(by[c].pop())
        i += 1
    rng.shuffle(out)
    return out


def write_tab(wb, title, question, heads, data):
    t = wb.create_sheet(title)
    for c, h in enumerate(heads, 1):
        t.cell(1, c, h)
    style_header(t, 1, len(heads))
    t.cell(1, LABEL_COL).comment = Comment(question + "  yes / no / skip", "pipeline")
    dv = DataValidation(type="list", formula1='"yes,no,skip"', allow_blank=True, showErrorMessage=True,
                        error="Choose yes, no or skip (or leave empty).")
    t.add_data_validation(dv)
    for i, vals in enumerate(data, 2):
        for c, v in enumerate(vals, 1):
            if isinstance(v, str):
                v = ILLEGAL_CHARACTERS_RE.sub(" ", v)
            cell = t.cell(i, c, v)
            cell.font = Font(name=FONT, size=10, bold=(c == 3), color="808080" if c >= 8 else "000000")
            cell.alignment = Alignment(vertical="top", wrap_text=c in (2, 4, 5, 7),
                                       horizontal="right" if c == 2 else None)
            cell.border = Border(bottom=THIN)
        t.cell(i, 3).fill = WORD_FILL
        t.cell(i, LABEL_COL).fill = INPUT_FILL
        t.cell(i, 7).fill = INPUT_FILL
        dv.add(t.cell(i, LABEL_COL))
    for c, w in enumerate([9, 38, 14, 38, 55, 9, 24, 14, 7, 12], 1):
        t.column_dimensions[t.cell(1, c).column_letter].width = w
    t.freeze_panes = "C2"
    return len(data)


def export(a):
    rng = random.Random(a.seed)
    web = {p["ref"]: p["web"] for p in read_jsonl(DERIVED / "bible_pairs.jsonl")}
    wb = Workbook()
    ws = wb.active
    ws.title = "Instructions"
    tabs = []

    jnt_data = {}
    for f, title in JNT_TABS.items():
        rows = [r for r in csv.DictReader(open(ANN / f"jnt_{f}.csv", encoding="utf-8")) if r["gold"].strip()]
        data = []
        for r in stratified(rows, a.per_tab, rng):
            b, w, af = split_context(r["context"])
            data.append([r["ref"], b, w, af, web.get(r["ref"], ""), "", "", r["ref"], int(r["word_idx"]), "jnt"])
        jnt_data[f] = data
    var_data = {}
    for kind in VARIETY_TABS:
        pool = []
        for key in VARIETY_SETS:
            for r in csv.DictReader(open(ANN / f"variety_{key}.csv", encoding="utf-8")):
                if r["kind"] == kind and r["gold"].strip():
                    pool.append({**r, "set": key})
        data = []
        for r in stratified(pool, a.per_tab, rng):
            b, w, af = split_context(r["context"])
            data.append([r["ref"].split(":")[1], b, w, af, r["sentence"], "", "", r["ref"], int(r["word_idx"]), r["set"]])
        var_data[kind] = data

    # instructions
    ws["A1"] = "Annotation: second opinion on a small sample"
    ws["A1"].font = Font(name=FONT, bold=True, size=16, color=PURPLE)
    ws["A3"], ws["A4"] = "Annotator", "Date"
    for c in ("B3", "B4"):
        ws[c].fill = INPUT_FILL
    lines = [
        ("What to do", True),
        ("Each tab lists words in context. The word to judge is highlighted in the WORD column. Choose yes, no or "
         "skip in the yellow LABEL column (drop-down). Leave it empty if you are unsure, and add a note if "
         "something is odd.", False),
        ("Judge meaning, not form. Many words were found automatically and will not be what the tab asks about: "
         "answer no or skip for those. These answers matter as much as the yes ones.", False),
        ("Only edit the yellow cells. Do not sort, delete or add rows, and do not edit the grey ID columns.", False),
        ("The first three tabs are from the Jamaican New Testament. The English column gives the World English "
         "Bible verse for orientation; the translations are free, so do not expect word-for-word matches.", False),
        ("The last two tabs are from interviews and oral histories (African American English and spoken Jamaican "
         "English). Some come from 1930s narratives written down in eye-dialect spelling (dey, wuz): judge what "
         "the speaker most likely said.", False),
        ("Confidential: the texts may only be used for research, and the Jamaican New Testament is © 2012 Bible "
         "Society of the West Indies. Please do not share or upload this file.", True),
    ]
    r = 6
    for t, bold in lines:
        ws.cell(r, 1, t).font = Font(name=FONT, bold=bold, size=12 if t == "What to do" else 10)
        ws.cell(r, 1).alignment = Alignment(wrap_text=True)
        r += 1
    r += 1
    for f, title in JNT_TABS.items():
        tabs.append((title, len(jnt_data[f])))
        ws.cell(r, 1, f"Tab '{title}'").font = Font(name=FONT, bold=True, size=12, color=PURPLE)
        ws.cell(r + 1, 1, QUESTIONS[("jnt", f)]).font = Font(name=FONT, italic=True)
        r += 2
        for g in GUIDE["jnt"][f]:
            ws.cell(r, 1, g).font = Font(name=FONT)
            ws.cell(r, 1).alignment = Alignment(wrap_text=True)
            r += 1
        r += 1
    for kind, (title, q, guide) in VARIETY_TABS.items():
        tabs.append((title, len(var_data[kind])))
        ws.cell(r, 1, f"Tab '{title}'").font = Font(name=FONT, bold=True, size=12, color=PURPLE)
        ws.cell(r + 1, 1, q).font = Font(name=FONT, italic=True)
        r += 2
        for g in guide:
            ws.cell(r, 1, g).font = Font(name=FONT)
            ws.cell(r, 1).alignment = Alignment(wrap_text=True)
            r += 1
        r += 1
    ws.cell(r, 1, "Progress").font = Font(name=FONT, bold=True, size=12)
    r += 1
    for c, h in enumerate(["Tab", "Rows", "Labelled", "Done"], 1):
        ws.cell(r, c, h)
    style_header(ws, r, 4)
    r += 1
    for title, n in tabs:
        ws.cell(r, 1, title).font = Font(name=FONT)
        ws.cell(r, 2, n).font = Font(name=FONT)
        ws.cell(r, 3, f"=COUNTA('{title}'!F2:F{n + 1})").font = Font(name=FONT)
        ws.cell(r, 4, f"=IF(B{r}=0,0,C{r}/B{r})").number_format = "0%"
        r += 1
    ws.column_dimensions["A"].width = 110
    for c in "BCD":
        ws.column_dimensions[c].width = 14

    jh = ["Verse", "Before", "WORD", "After", "English (WEB)", "LABEL", "Note", "ID: ref", "ID: word", "ID: set"]
    vh = ["Sentence", "Before", "WORD", "After", "Whole sentence", "LABEL", "Note", "ID: ref", "ID: word", "ID: set"]
    for f, title in JNT_TABS.items():
        write_tab(wb, title, QUESTIONS[("jnt", f)], jh, jnt_data[f])
    for kind, (title, q, _) in VARIETY_TABS.items():
        write_tab(wb, title, q, vh, var_data[kind])
    wb.calculation.fullCalcOnLoad = True
    out = ANN / "annotate_second.xlsx"
    wb.save(out)
    print(f"{out}: " + ", ".join(f"{t} {n}" for t, n in tabs) + f" ({sum(n for _, n in tabs)} rows)")


def kappa(pairs, cats):
    n = len(pairs)
    if not n:
        return None
    po = sum(x == y for x, y in pairs) / n
    pa, pb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(pa[c] * pb[c] for c in cats) / n ** 2
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def import_(a):
    first = {}
    for f in JNT_TABS:
        for r in csv.DictReader(open(ANN / f"jnt_{f}.csv", encoding="utf-8")):
            if r["gold"].strip():
                first[(JNT_TABS[f], r["ref"], int(r["word_idx"]))] = r["gold"].strip()
    for key in VARIETY_SETS:
        for r in csv.DictReader(open(ANN / f"variety_{key}.csv", encoding="utf-8")):
            if r["kind"] in VARIETY_TABS and r["gold"].strip():
                first[(VARIETY_TABS[r["kind"]][0], r["ref"], int(r["word_idx"]))] = r["gold"].strip()
    report = {"note": "second annotator vs first annotator (CA) on a stratified subset", "tabs": {}}
    for path in a.files:
        wb = load_workbook(path, data_only=True)
        report["second_annotator"] = wb["Instructions"]["B3"].value or ""
        for title in list(JNT_TABS.values()) + [t for t, _, _ in VARIETY_TABS.values()]:
            if title not in wb.sheetnames:
                continue
            pairs = []
            for row in wb[title].iter_rows(min_row=2, values_only=True):
                lab = (row[LABEL_COL - 1] or "").strip().lower()
                if not (row[7] and lab):
                    continue
                if lab not in TO_GOLD:
                    raise SystemExit(f"{path} {title}: unknown label {lab!r} at {row[7]}")
                g1 = first.get((title, row[7], int(row[8])))
                if g1 is not None:
                    pairs.append((g1, TO_GOLD[lab]))
            binary = [(x, y) for x, y in pairs if x in "01" and y in "01"]
            report["tabs"][title] = {
                "n": len(pairs), "agreement": round(sum(x == y for x, y in pairs) / len(pairs), 3) if pairs else None,
                "kappa_3way": kappa(pairs, "01x"), "n_binary": len(binary),
                "agreement_binary": round(sum(x == y for x, y in binary) / len(binary), 3) if binary else None,
                "kappa_binary": kappa(binary, "01"),
                "confusion": {f"{x}|{y}": c for (x, y), c in sorted(Counter(pairs).items())}}
    write_json("specimens/agreement_report.json", report)
    print(f"{'tab':18s} {'n':>4s} {'agree':>6s} {'kappa':>6s} | {'yes/no only':>11s} {'kappa':>6s}")
    for t, r in report["tabs"].items():
        print(f"{t:18s} {r['n']:4d} {r['agreement'] or 0:6.0%} {str(r['kappa_3way']):>6s} | "
              f"{r['agreement_binary'] or 0:11.0%} {str(r['kappa_binary']):>6s}")
    print("-> specimens/agreement_report.json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--per-tab", type=int, default=20)
    e.add_argument("--seed", type=int, default=0)
    i = sub.add_parser("import")
    i.add_argument("files", nargs="+")
    a = ap.parse_args()
    export(a) if a.cmd == "export" else import_(a)


if __name__ == "__main__":
    main()
