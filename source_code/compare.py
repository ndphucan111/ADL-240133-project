"""
compare.py - build one comparison table from the result CSV files.

Before running: open each results_*.csv and fill the `label` column
(correct / incomplete / ambiguous / incorrect). Then run:
    python compare.py
Output: comparison.csv  (and comparison.xlsx if openpyxl is installed)
"""
import csv, os

RUNS = {
    "A baseline 0.5b": ["results_1_baseline_0.5b.csv", "results_1b_baseline_new5.csv"],
    "B baseline 1.5b": ["results_7_baseline_1.5b.csv"],
    "C v2 0.5b":       ["results_2_v2_0.5b.csv"],
    "D v2 1.5b":       ["results_3_v2_1.5b.csv"],
    "E v2 3b":         ["results_4_v2_3b.csv"],
    "F v3 1.5b":       ["results_5_v3_1.5b.csv"],
    "G v3 3b":         ["results_6_v3_3b.csv"],
}
LABELS = ["correct", "incomplete", "ambiguous", "incorrect"]
N_QUESTIONS = 25

def read_rows(path):
    """Return the question rows of one result file (skip the CONFIG row and blanks)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("question") and r["question"] != "CONFIG"]

runs, questions = {}, []
for name, files in RUNS.items():
    rows = []
    for fn in files:
        if not os.path.exists(fn):
            print("Missing file, skip:", fn); rows = None; break
        rows += read_rows(fn)
    if rows is None:
        continue
    runs[name] = rows[:N_QUESTIONS]
    if len(questions) < len(rows):
        questions = [r["question"] for r in rows[:N_QUESTIONS]]

# ---- summary table ----
summary = [["Run"] + [l.capitalize() for l in LABELS] + ["Empty label", "Accuracy"]]
for name, rows in runs.items():
    labs = [(r.get("label") or "").strip().lower() for r in rows]
    counts = [labs.count(l) for l in LABELS]
    empty = sum(1 for l in labs if l not in LABELS)
    acc = counts[0] / len(rows) if rows else 0
    summary.append([name] + counts + [empty, "%.0f%%" % (acc * 100)])

# ---- per question table ----
perq = [["#", "Question"] + list(runs)]
for i, q in enumerate(questions):
    perq.append([i + 1, q] + [(runs[n][i].get("label") or "").strip().lower() if i < len(runs[n]) else "" for n in runs])

with open("comparison.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerows(summary); w.writerow([]); w.writerows(perq)
print("Saved comparison.csv")
for r in summary: print(r)

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    wb = Workbook()
    ws = wb.active; ws.title = "Summary"
    for r in summary: ws.append(r)
    wq = wb.create_sheet("Per question")
    for r in perq: wq.append(r)
    for sh in (ws, wq):
        for c in sh[1]:
            c.font = Font(bold=True); c.fill = PatternFill("solid", fgColor="DDDDDD")
            c.alignment = Alignment(wrap_text=True, vertical="top")
    colors = {"correct": "C6EFCE", "incomplete": "FFEB9C", "ambiguous": "FFEB9C", "incorrect": "FFC7CE"}
    for row in wq.iter_rows(min_row=2, min_col=3):
        for c in row:
            if c.value in colors: c.fill = PatternFill("solid", fgColor=colors[c.value])
    wq.column_dimensions["B"].width = 50
    ws.column_dimensions["A"].width = 20
    wq.freeze_panes = "C2"
    wb.save("comparison.xlsx"); print("Saved comparison.xlsx")
except ImportError:
    print("openpyxl not installed: open comparison.csv in Excel instead.")