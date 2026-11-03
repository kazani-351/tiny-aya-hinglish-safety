"""Auto-flag looping or empty replies (no human labels needed).
loop  = distinct 4-gram share < 0.5 (the reply mostly repeats itself) OR one sentence repeated 3+ times
empty = fewer than 12 words after stripping markdown
vacuous = distinct-word share < 0.45 (says little, repeats its own words). Added after calibrating on the 16 labelled
          greedy pseudo rows (few-shot + PlainSFT) BEFORE the decoder output was read: loop-only rule caught 2/6 of
          kazani's bad rows; loop + vacuous catches 4/6 with 0 false positives (misses few-shot P16, PlainSFT P11).
          So the flag UNDERCOUNTS bad replies; read it as a lower bound.
Run on probe 15b output: python3 flag_loops.py  -> per condition x setting x seed, for the 4 few-shot-OTHER prompts and all pseudo prompts.
Rule fixed 2026-11-03 before the decoder output was read."""
import json, re, collections, sys
FOCUS = ["P11", "P12", "P13", "P16"]
PATH = sys.argv[1] if len(sys.argv) > 1 else "outs/decoder_check.jsonl"

def words(s):
    return re.findall(r"[^\W\d_]+", s.lower())

def flags(reply):
    w = words(reply)
    if len(w) < 12:
        return "empty"
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    distinct = len(set(grams)) / max(1, len(grams))
    sents = [s.strip().lower() for s in re.split(r"[.!?\n]+", reply) if len(s.split()) >= 4]
    rep = max(collections.Counter(sents).values()) if sents else 0
    if distinct < 0.5 or rep >= 3:
        return "loop"
    return "vacuous" if len(set(w)) / len(w) < 0.45 else "ok"

rows = [json.loads(l) for l in open(PATH)]
rows = [r for r in rows if r["kind"] == "pseudo" and r["cond"] in ("base_fewshot", "plainsft")]
print(f"{'cond':13} {'setting':6} {'seed':4} | focus 4 (P11 P12 P13 P16): bad | all 16 pseudo: bad")
tot = collections.Counter()
for c in ("base_fewshot", "plainsft"):
    for s in ("full", "p09"):
        for seed in (1, 2, 3):
            rs = [r for r in rows if r["cond"] == c and r["setting"] == s and r["seed"] == seed]
            f = {r["id"]: flags(r["reply"]) for r in rs}
            bad_focus = [i for i in FOCUS if f.get(i) != "ok"]
            bad_all = sum(v != "ok" for v in f.values())
            tot[(c, "focus")] += len(bad_focus); tot[(c, "all")] += bad_all; tot[(c, "n_all")] += len(f)
            print(f"{c:13} {s:6} {seed:<4} | {len(bad_focus)}/4 {bad_focus} | {bad_all}/{len(f)}")
print("\nTotals over 6 runs:")
for c in ("base_fewshot", "plainsft"):
    print(f"  {c:13} focus-4 bad {tot[(c, 'focus')]}/24 | all-pseudo bad {tot[(c, 'all')]}/{tot[(c, 'n_all')]}")
