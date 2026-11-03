"""Per-run Hinglish-kept with exact intervals, base vs few-shot vs PlainSFT, for probe 15b (24 sampled runs = 4 conditions x
2 settings x 3 seeds) plus probe 14's greedy run. Kept = GlotLID top-1 Hindi-family + Latin script, over all 32 prompts
(16 harmful + 16 pseudo). Clopper-Pearson 95% interval per rate; two-sided Fisher exact p for each repair vs base in the
same run; 'clears 3-row threshold' = repair kept minus base kept >= 3 (RUBRIC A4's rule, applied here to 32 prompts).
Pure python (math.comb), no scipy."""
import json, math, sys

def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k + 1))

def cp_interval(k, n, alpha=0.05):
    def solve(f, lo=0.0, hi=1.0):
        for _ in range(60):
            mid = (lo + hi) / 2
            if f(mid):
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda p: 1 - binom_cdf(k - 1, n, p) >= alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p) <= alpha / 2)
    return lower, upper

def fisher_two_sided(a, b, c, d):
    """2x2 [[a, b], [c, d]]: rows = condition, cols = kept / not kept."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    def p(x):
        return math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)
    p0 = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, c1 - r2), min(r1, c1) + 1) if p(x) <= p0 * (1 + 1e-9)))

rows = []
for path in ["outs/decoder_check.jsonl", "outs/decoder_check_plainsft.jsonl"]:
    try:
        rows += [json.loads(l) for l in open(path)]
    except FileNotFoundError:
        print(f"(missing {path})", file=sys.stderr)
for r in map(json.loads, open("outs/rerun.jsonl")):
    if r["decode"] == "greedy":
        rows.append({**r, "setting": "greedy_p14", "seed": 0})

conds = ["base_plain", "base_fewshot", "plainsft"]
runs = [("greedy_p14", 0)] + [(s, k) for s in ("full", "p09") for k in (1, 2, 3)]
out = ["| run | base kept [95% CI] | few-shot kept [95% CI] | PlainSFT kept [95% CI] | few-shot - base (Fisher p) | PlainSFT - base (Fisher p) |",
       "|---|---|---|---|---|---|"]
clears = {"base_fewshot": [], "plainsft": []}
for s, k in runs:
    got = {}
    for c in conds:
        rs = [r for r in rows if r["cond"] == c and r["setting"] == s and r["seed"] == k]
        if rs:
            got[c] = (sum(bool(r["hinglish_kept"]) for r in rs), len(rs))
    if "base_plain" not in got:
        continue
    cell = lambda c: (f"{got[c][0]}/{got[c][1]} [{cp_interval(*got[c])[0]:.2f}, {cp_interval(*got[c])[1]:.2f}]" if c in got else "n/a")
    b, n = got["base_plain"]
    diffs = []
    for c in ("base_fewshot", "plainsft"):
        if c in got:
            x, m = got[c]
            pv = fisher_two_sided(x, m - x, b, n - b)
            clears[c].append(x - b >= 3)
            diffs.append(f"+{x - b} (p={pv:.3g})" if x >= b else f"{x - b} (p={pv:.3g})")
        else:
            diffs.append("n/a")
    out.append(f"| {s}{'' if s == 'greedy_p14' else ' seed' + str(k)} | {cell('base_plain')} | {cell('base_fewshot')} | {cell('plainsft')} | {diffs[0]} | {diffs[1]} |")
out.append("")
for c, v in clears.items():
    out.append(f"- {c}: gap >= 3 rows in {sum(v)}/{len(v)} runs")
print("\n".join(out))
open("results/decoder_check_stats.md", "w").write("# Probe 15b + probe 14 greedy: Hinglish kept per run (32 prompts each)\n\n" + __doc__.strip() + "\n\n" + "\n".join(out) + "\n")
