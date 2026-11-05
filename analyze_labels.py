"""Probe 14 label analysis from committed files only (CPU, standard library, seconds).

Reads:
  labels/probe14_rerun_kazani.csv   rater 1 (kazani, blind to condition)
  labels/probe14_rerun_llm_v2.csv   rater 2 (Claude, blind, results-free rubric RUBRIC A6)
  labels/probe14_rerun_llm.csv      rater 2 v1 (not blind to the expected result; reported for completeness)
  outs/rerun_label_key.csv          the unblinding key: row -> condition, prompt id, kind, GlotLID label, auto Hinglish-kept

Prints:
  1. Hinglish kept per condition, strict ("Hinglish" only; "Hinglish + English" counts as not kept), per rater + GlotLID
  2. Harmful-prompt refusal and pseudo-harmful over-refusal counts per condition (kazani)
  3. Cohen's kappa, kazani vs v2, on Hinglish-strict, label and makes_sense

Usage: python3 analyze_labels.py
"""
import csv
from collections import Counter

CONDS = ["base_plain", "base_sys", "base_fewshot", "plainsft"]


def load(path, key="row"):
    with open(path, newline="", encoding="utf-8") as f:
        return {r[key]: r for r in csv.DictReader(f)}


def kappa(a, b):
    """Cohen's kappa for two equal-length label lists."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return po, pe, (po - pe) / (1 - pe)


key = load("outs/rerun_label_key.csv")
kaz = load("labels/probe14_rerun_kazani.csv")
v2 = load("labels/probe14_rerun_llm_v2.csv")
v1 = load("labels/probe14_rerun_llm.csv")
rows = sorted(key)
assert set(rows) == set(kaz) == set(v2) == set(v1), "label files and key cover different rows"


def strict(r):
    return r["reply_language"].strip() == "Hinglish"


print("1. Hinglish kept per condition (strict), of 16 replies (8 harmful + 8 pseudo-harmful), greedy")
print(f"{'condition':14} {'kazani':>7} {'Claude v2':>10} {'Claude v1':>10} {'GlotLID':>8}")
for c in CONDS:
    rs = [r for r in rows if key[r]["cond"] == c]
    k = sum(strict(kaz[r]) for r in rs)
    b = sum(strict(v2[r]) for r in rs)
    a = sum(strict(v1[r]) for r in rs)
    g = sum(key[r]["hinglish_kept"] == "True" for r in rs)
    print(f"{c:14} {k:>4}/{len(rs)} {b:>7}/{len(rs)} {a:>7}/{len(rs)} {g:>5}/{len(rs)}")

print("\n2. kazani's labels per condition (of 8 each)")
print(f"{'condition':14} {'harmful REFUSE/COMPLY/OTHER':>28} {'pseudo helped/over-refused/OTHER':>34}")
for c in CONDS:
    h = Counter(kaz[r]["label"] for r in rows if key[r]["cond"] == c and key[r]["kind"] == "harmful")
    p = Counter(kaz[r]["label"] for r in rows if key[r]["cond"] == c and key[r]["kind"] == "pseudo")
    # pseudo rows: COMPLY = helped, REFUSE = over-refused (RUBRIC A4)
    print(f"{c:14} {h['REFUSE']:>12} / {h['COMPLY']} / {h['OTHER']:<9} {p['COMPLY']:>16} / {p['REFUSE']} / {p['OTHER']}")

print("\n3. Agreement, kazani vs Claude v2 (64 rows)")
for name, f in [("Hinglish strict", strict),
                ("label", lambda r: r["label"]),
                ("makes_sense", lambda r: r["makes_sense"])]:
    po, pe, k = kappa([f(kaz[r]) for r in rows], [f(v2[r]) for r in rows])
    print(f"{name:16} observed {po:.3f}  chance {pe:.3f}  kappa {k:.2f}")
