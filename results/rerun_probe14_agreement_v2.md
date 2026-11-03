# Probe 14 agreement, kazani vs v2 second rater (RUBRIC A6; v2 labels committed ea81c72 before comparison)
v2 = fresh-context Claude subagent, shown only outs/rubric_for_rater_v2.md (rubric core + A4, no results, no condition names, no probe-4 numbers) and the blind sheet, with a neutral prompt. v1 (A5) saw the expected result in A5's text: reported here only for comparison, never called blind.

| Measure (64 rows) | observed | chance | kappa | marginals (kazani, v2) |
|---|---|---|---|---|
| label | 0.812 | 0.410 | 0.68 | COMPLY 31/35, OTHER 11/4, REFUSE 22/25 |
| makes_sense | 0.750 | 0.396 | 0.59 | no 8/9, partly 36/28, yes 20/27 |
| Hinglish strict | 0.953 | 0.633 | 0.87 | not 16/15, Hinglish 48/49 |

- The base rate is high (chance agreement 0.63 on Hinglish-or-not), and kappa 0.87 is well above it. Per condition: few-shot is 16/16 for every rater, so kappa is undefined there; PlainSFT 14 vs 16 gives kappa 0 (the paradox: one rater has no "not" rows). The per-condition counts are the honest view:

| Condition | kazani | v2 | v1 (not blind) | GlotLID |
|---|---|---|---|---|
| base_plain | 9/16 | 8/16 | 9/16 | 6/16 |
| base_sys | 9/16 | 9/16 | 10/16 | 11/16 |
| base_fewshot | 16/16 | 16/16 | 16/16 | 16/16 |
| plainsft | 14/16 | 16/16 | 16/16 | 16/16 |

- kazani's 6 unsure rows: label agreement 2/6 (lower than v1's 4/6), language 6/6.
- Leak effect: v1 vs v2 differ on 4/64 labels and 2/64 language calls. The leak barely moved the labels.
- v2 behaviour per condition: harmful refused 5-7/8, complied 0-2/8; pseudo helped 7-8/8, over-refused 0/8 everywhere. Same picture as kazani's.
- Independence limits, stated plainly: v2 is the same model family as the session that ran the study; it is a second reading, not an independent human panel.
