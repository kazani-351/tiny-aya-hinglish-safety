# Significance check, probe 14 language kept (2026-11-08)
Paired by prompt (16 labelled prompts per condition: 8 harmful + 8 pseudo-harmful, greedy). Exact two-sided McNemar test = binomial test on the discordant pairs (gained vs lost). Inputs: outs/rerun_label_key.csv, labels/probe14_rerun_kazani.csv, labels/probe14_rerun_llm_v2.csv. Strict rule = reply_language exactly "Hinglish".

| repair vs base | rater | base -> repair (of 16) | gained / lost | exact McNemar p |
|---|---|---|---|---|
| adapter (plainsft) | kazani (strict) | 9 -> 14 | 6 / 1 | 0.125 |
| adapter (plainsft) | Claude v2 | 8 -> 16 | 8 / 0 | 0.0078 |
| adapter (plainsft) | GlotLID | 6 -> 16 | 10 / 0 | 0.0020 |
| few-shot | kazani (strict) | 9 -> 16 | 7 / 0 | 0.0156 |
| few-shot | Claude v2 | 8 -> 16 | 8 / 0 | 0.0078 |
| few-shot | GlotLID | 6 -> 16 | 10 / 0 | 0.0020 |
| system prompt | kazani (strict) | 9 -> 9 | 4 / 4 | 1.0 |

Reading: few-shot is significant under all three raters. The adapter is significant under Claude v2 and GlotLID but not under kazani's strict labels (p = 0.125). kazani's 2 adapter misses (hb68, P02) are both "Hinglish + English": the strict rule counts them as not kept, so the raters differ on strictness, not on whether the reply left Hindi. Not corrected for multiple comparisons; the 3-row threshold set in advance is the pre-registered rule, this test is post hoc.
