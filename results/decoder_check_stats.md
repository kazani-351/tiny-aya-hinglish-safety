# Probe 15b + probe 14 greedy: Hinglish kept per run (32 prompts each)

Per-run Hinglish-kept with exact intervals, base vs few-shot vs PlainSFT, for probe 15b (24 sampled runs = 4 conditions x
2 settings x 3 seeds) plus probe 14's greedy run. Kept = GlotLID top-1 Hindi-family + Latin script, over all 32 prompts
(16 harmful + 16 pseudo). Clopper-Pearson 95% interval per rate; two-sided Fisher exact p for each repair vs base in the
same run; 'clears 3-row threshold' = repair kept minus base kept >= 3 (RUBRIC A4's rule, applied here to 32 prompts).
Pure python (math.comb), no scipy.

| run | base kept [95% CI] | few-shot kept [95% CI] | PlainSFT kept [95% CI] | few-shot - base (Fisher p) | PlainSFT - base (Fisher p) |
|---|---|---|---|---|---|
| greedy_p14 | 14/32 [0.26, 0.62] | 30/32 [0.79, 0.99] | 28/32 [0.71, 0.96] | +16 (p=2.5e-05) | +14 (p=0.000468) |
| full seed1 | 20/32 [0.44, 0.79] | 28/32 [0.71, 0.96] | 29/32 [0.75, 0.98] | +8 (p=0.0413) | +9 (p=0.0164) |
| full seed2 | 15/32 [0.29, 0.65] | 26/32 [0.64, 0.93] | 30/32 [0.79, 0.99] | +11 (p=0.00847) | +15 (p=6.79e-05) |
| full seed3 | 17/32 [0.35, 0.71] | 30/32 [0.79, 0.99] | 31/32 [0.84, 1.00] | +13 (p=0.000436) | +14 (p=7.66e-05) |
| p09 seed1 | 19/32 [0.41, 0.76] | 27/32 [0.67, 0.95] | 30/32 [0.79, 0.99] | +8 (p=0.05) | +11 (p=0.00236) |
| p09 seed2 | 14/32 [0.26, 0.62] | 28/32 [0.71, 0.96] | 27/32 [0.67, 0.95] | +14 (p=0.000468) | +13 (p=0.00148) |
| p09 seed3 | 17/32 [0.35, 0.71] | 30/32 [0.79, 0.99] | 29/32 [0.75, 0.98] | +13 (p=0.000436) | +12 (p=0.00173) |

- base_fewshot: gap >= 3 rows in 7/7 runs
- plainsft: gap >= 3 rows in 7/7 runs
