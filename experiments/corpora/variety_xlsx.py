"""Annotation workbooks for the variety-specific constructions, and import of the labels.

  python experiments/corpora/variety_xlsx.py export [--only coraal wpa ...]
      data/annotation/variety_<collection>.csv -> data/annotation/annotate_variety_<collection>.xlsx
  python experiments/corpora/variety_xlsx.py import data/annotation/annotate_variety_coraal.xlsx ...
      labels -> variety_<collection>.csv (gold), and merged into data/derived/sample_<collection>.jsonl:
        plural / past / progressive -> labels[feature][word] = 1/0, source "gold", marking "unmarked"
        habitual be, copula absence, 3rd-person -s -> extra_labels (not probed against WEB directions)

Rows are candidates found automatically (variety_candidates.py), so many will be "no" or
"skip": that is expected, and the "no" answers are as useful as the "yes" ones.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bible"))
sys.path.insert(0, str(HERE))
from openpyxl import Workbook, load_workbook  # noqa: E402
from openpyxl.comments import Comment  # noqa: E402
from openpyxl.styles import Alignment, Border, Font  # noqa: E402
from openpyxl.worksheet.datavalidation import DataValidation  # noqa: E402
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE  # noqa: E402

from annotation_xlsx import FONT, HEAD_FILL, INPUT_FILL, PURPLE, THIN, TO_GOLD, WORD_FILL, style_header  # noqa: E402
from common import DERIVED, read_jsonl, write_jsonl  # noqa: E402
from tokenisation_profile import LABEL  # noqa: E402

ANN = Path("data/annotation")
PRIORITY = ["coraal", "wpa", "ice-ja-spoken"]
TABS = {
    "plural_unmarked": ("Plural, no -s", "plural",
                        "Does the highlighted noun refer to MORE THAN ONE thing?",
                        ["yes: plural in meaning although it has no -s (two [mile], them [boy], di [bwai] dem).",
                         "no: it refers to one thing, or it is a mass noun (all [day], some [money]).",
                         "skip: not a noun, or a fixed expression (o'clock), or you cannot tell."]),
    "past_unmarked": ("Past, no -ed", "past",
                      "Does the highlighted verb describe a PAST event or state?",
                      ["yes: past in meaning although the verb has no past form (he [come] up there "
                       "yesterday; Jamaican did [go]).",
                       "no: present, future, habitual or general (you [know], they [grow] up thinking).",
                       "skip: not a verb, or you cannot tell from the sentence."]),
    "progressive_nobe": ("Progressive, no be", "progressive",
                         "Does the highlighted verb describe an action IN PROGRESS at that time?",
                         ["yes: ongoing action without is/are/was (she [going] to fifth grade; Jamaican im a [go]).",
                          "no: not in progress, or the -ing word is a noun or adjective ([growing] up as a phrase "
                          "about childhood counts as no).",
                          "skip: not a verb, or you cannot tell."]),
    "habitual_be": ("Habitual be", "habitual",
                    "Does BE here say that something happens REGULARLY or habitually?",
                    ["yes: habitual be (Sometimes he [be] on the movies; Everybody [be] happy).",
                     "no: another be: subjunctive (whether it [be] true), imperative ([Be] careful), "
                     "after a left-out modal, or a quotation.",
                     "skip: you cannot tell."]),
    "copula_absence": ("No copula", "copula_absence",
                       "Is a form of BE (is / are / was) LEFT OUT just before the highlighted word?",
                       ["yes: be is missing (he [mean]; she [gonna] call; they [in] the house).",
                        "no: nothing is missing (the word is a verb, or be is there elsewhere).",
                        "skip: a fragment or false start where you cannot tell."]),
    "agreement_3sg": ("No 3rd-person -s", "agreement_unmarked",
                      "Is this a PRESENT-tense verb with a he / she / it subject, used WITHOUT -s?",
                      ["yes: he [go], she [have], it [make] (present meaning, no -s).",
                       "no: it is past in meaning (then that's a Past case: note it), or the subject is not "
                       "he/she/it, or the verb is not finite.",
                       "skip: you cannot tell."]),
}
CORE = {"plural", "past", "progressive"}


def sheet_rows(key):
    with open(ANN / f"variety_{key}.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def export(a):
    keys = a.only or sorted(p.stem.replace("variety_", "") for p in ANN.glob("variety_*.csv"))
    for key in keys:
        rows = sheet_rows(key)
        wb = Workbook()
        ws = wb.active
        ws.title = "Instructions"
        ws["A1"] = f"Annotation: {LABEL.get(key, key)}"
        ws["A1"].font = Font(name=FONT, bold=True, size=16, color=PURPLE)
        ws["A3"], ws["A4"] = "Annotator", "Date started"
        for c in ("B3", "B4"):
            ws[c].fill = INPUT_FILL
        lines = [
            ("What to do", True),
            ("Each tab lists words that MIGHT be a non-standard construction. The word to judge is "
             "highlighted in the WORD column; the full sentence is next to it.", False),
            ("Choose yes, no or skip in the yellow LABEL column (drop-down). Many rows will be 'no' or "
             "'skip': they were found automatically, and those answers matter as much as the 'yes' ones.", False),
            ("Judge meaning, not form. Only edit the yellow cells; do not sort, delete or add rows.", False),
        ]
        if key == "wpa":
            lines.append(("Note: these 1930s narratives were mostly written down by white interviewers in "
                          "eye-dialect spelling (dey, wuz). Judge what the speaker most likely said, and use "
                          "the Note column when the spelling itself seems to be the issue.", False))
        if key in PRIORITY:
            lines.append(("This collection is a first priority.", True))
        r = 6
        for t, bold in lines:
            ws.cell(r, 1, t).font = Font(name=FONT, bold=bold, size=12 if bold else 10)
            ws.cell(r, 1).alignment = Alignment(wrap_text=True)
            r += 1
        r += 1
        present = [k for k in TABS if any(x["kind"] == k for x in rows) and (not a.kinds or k in a.kinds)]
        for k in present:
            title, _, q, guide = TABS[k]
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
        for k in present:
            title = TABS[k][0]
            n = sum(1 for x in rows if x["kind"] == k)
            ws.cell(r, 1, title).font = Font(name=FONT)
            ws.cell(r, 2, n).font = Font(name=FONT)
            ws.cell(r, 3, f"=COUNTA('{title}'!F2:F{n + 1})").font = Font(name=FONT)
            ws.cell(r, 4, f"=IF(B{r}=0,0,C{r}/B{r})").number_format = "0%"
            r += 1
        ws.column_dimensions["A"].width = 110
        for c in "BCD":
            ws.column_dimensions[c].width = 14

        for k in present:
            title, _, q, _ = TABS[k]
            t = wb.create_sheet(title)
            heads = ["Sentence", "Before", "WORD", "After", "Whole sentence", "LABEL", "Note", "ID: ref", "ID: word"]
            for c, h in enumerate(heads, 1):
                t.cell(1, c, h)
            style_header(t, 1, len(heads))
            t.cell(1, 6).comment = Comment(q + "  yes / no / skip", "pipeline")
            dv = DataValidation(type="list", formula1='"yes,no,skip"', allow_blank=True,
                                showErrorMessage=True, error="Choose yes, no or skip (or leave empty).")
            t.add_data_validation(dv)
            sel = [x for x in rows if x["kind"] == k]
            for i, x in enumerate(sel, 2):
                words = x["context"].split(" ")
                j = next(n for n, w in enumerate(words) if w.startswith("[") and w.endswith("]"))
                back = {"1": "yes", "0": "no", "x": "skip"}.get(x.get("gold", ""), "")
                vals = [x["ref"].split(":")[1], " ".join(words[:j]), words[j][1:-1], " ".join(words[j + 1:]),
                        x["sentence"], back, x.get("note", ""), x["ref"], int(x["word_idx"])]
                for c, v in enumerate(vals, 1):
                    if isinstance(v, str):
                        v = ILLEGAL_CHARACTERS_RE.sub(" ", v)
                    cell = t.cell(i, c, v)
                    cell.font = Font(name=FONT, size=10, bold=(c == 3), color="808080" if c >= 8 else "000000")
                    cell.alignment = Alignment(vertical="top", wrap_text=c in (2, 4, 5, 7),
                                               horizontal="right" if c == 2 else None)
                    cell.border = Border(bottom=THIN)
                t.cell(i, 3).fill = WORD_FILL
                t.cell(i, 6).fill = INPUT_FILL
                t.cell(i, 7).fill = INPUT_FILL
                dv.add(t.cell(i, 6))
            for c, w in enumerate([9, 38, 14, 38, 55, 9, 24, 14, 7], 1):
                t.column_dimensions[t.cell(1, c).column_letter].width = w
            t.freeze_panes = "C2"
        wb.calculation.fullCalcOnLoad = True
        out = ANN / f"annotate_variety_{key}.xlsx"
        wb.save(out)
        print(f"{out}: " + ", ".join(f"{TABS[k][0]} {sum(1 for x in rows if x['kind'] == k)}" for k in present))


def import_(a):
    by_title = {v[0]: k for k, v in TABS.items()}
    for path in a.files:
        key = Path(path).stem.replace("annotate_variety_", "").split("_annotated")[0]
        wb = load_workbook(path)
        got = {}
        for title, k in by_title.items():
            if title not in wb.sheetnames:
                continue
            for row in wb[title].iter_rows(min_row=2, values_only=True):
                lab = (row[5] or "").strip().lower()
                if row[7] and lab:
                    if lab not in TO_GOLD:
                        raise SystemExit(f"{path} {title}: unknown label {lab!r} at {row[7]}")
                    got[(row[7], int(row[8]), k)] = (TO_GOLD[lab], (row[6] or "").strip())
        rows = sheet_rows(key)
        for x in rows:
            g = got.get((x["ref"], int(x["word_idx"]), x["kind"]))
            if g:
                x["gold"], x["note"] = g
        with open(ANN / f"variety_{key}.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        # merge into the sample
        sp = DERIVED / f"sample_{key}.jsonl"
        sample = read_jsonl(sp)
        by_ref = {s["ref"]: s for s in sample}
        n_core = n_extra = 0
        for x in rows:
            if x["gold"] not in ("0", "1"):
                continue
            s = by_ref.get(x["ref"])
            if s is None:
                continue
            feat, wi, y = TABS[x["kind"]][1], x["word_idx"], int(x["gold"])
            if feat in CORE:
                s["labels"][feat][wi] = y
                s["sources"][feat][wi] = "gold"
                s.setdefault("marking", {}).setdefault(feat, {})[wi] = "unmarked"
                n_core += 1
            else:
                s.setdefault("extra_labels", {}).setdefault(feat, {})[wi] = y
                n_extra += 1
        write_jsonl(sp, sample)
        print(f"{path}: {sum(1 for x in rows if x['gold'])} labels -> variety_{key}.csv; "
              f"merged into sample_{key}.jsonl ({n_core} plural/past/progressive, {n_extra} other)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--only", nargs="*")
    e.add_argument("--kinds", nargs="*", choices=list(TABS), help="only these tabs (default all)")
    i = sub.add_parser("import")
    i.add_argument("files", nargs="+")
    a = ap.parse_args()
    export(a) if a.cmd == "export" else import_(a)


if __name__ == "__main__":
    main()
