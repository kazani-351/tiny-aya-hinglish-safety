# Devanagari script fidelity, probe 7 (2026-11-08 recount)
Source: outs/script_probe.jsonl (probes/script_probe.py, Nov 1). Unsteered Tiny Aya Global, the 16 harmful prompts in 3 forms, greedy + 3 sampling seeds (temperature 0.7) = 64 replies per form. Strict rule: GlotLID top label is a Hindi-family language AND in the prompt's script AND at least 50% of letters in that script (Devanagari for hi; Latin for hi_roman and hinglish). Automatic labels only: kazani did not hand-label these replies.

| prompt form | replies kept (strict) | prompts kept on all 4 decodes |
|---|---|---|
| Hindi, Devanagari | 64 / 64 | 16 / 16 |
| Hindi, Latin script | 42 / 64 | 5 / 16 |
| Hinglish | 25 / 64 | 4 / 16 |

Lowest Devanagari letter share among the 64 Devanagari replies: 0.56 (one reply mixes scripts; still mostly Devanagari).
