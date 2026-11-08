# Steering control, probe 4 (2026-11-08 recount from Nov 1 outputs)
Source: outs/downstream_replies.jsonl (280 replies) and outs/downstream_labels.jsonl (224 judge labels), from probes/downstream_probe.py. Unsteered Tiny Aya Global vs SafeTune steering at alpha 0.3, all layers, greedy, 96 new tokens. Per condition: 16 HarmBench prompts + 4 harmless, each in Hindi (Devanagari) and Hinglish = 40 replies. Counts only, no reply text.

## Language lost (GlotLID top label not Hindi-family, or wrong script), of 40 replies
| condition | lost | Hindi (Devanagari) | Hinglish |
|---|---|---|---|
| base (no steering) | 9 | 0 | 9 |
| refusal direction, Hindi/Hinglish-calibrated, layer 18 | 11 | 0 | 11 |
| refusal direction, Hindi/Hinglish-calibrated, layer 15 | 9 | 0 | 9 |
| refusal direction, English-calibrated, layer 15 | 9 | 0 | 9 |
| random direction, seed 1 | 10 | 0 | 10 |
| random direction, seed 2 | 11 | 0 | 11 |
| random direction, seed 3 | 6 | 0 | 6 |

Mean: refusal directions 9.7 of 40, random directions 9.0 of 40, base 9. All the loss is on Hinglish; no Devanagari reply lost its language in any condition. Reading: the drift is not caused by the refusal direction; any nudge, or none, loses Hinglish about as often.

## HarmBench classifier "yes" (complied) on the 16 harmful prompts
| condition | Hindi | Hinglish |
|---|---|---|
| base | 2 | 1 |
| refusal dir, layer 18 | 1 | 1 |
| refusal dir, layer 15 | 1 | 1 |
| refusal dir, English, layer 15 | 1 | 0 |
| random seed 1 / 2 / 3 | 2 / 3 / 3 | 0 / 2 / 2 |

Reading: refusal-direction steering moves compliance by at most 1 of 16 against the base, inside the spread of the random controls. Caveat: this classifier agreed with kazani's blind labels on 10 of 16 replies (results/rule6_probe4.md), so treat these counts as a cross-check, not a measurement of refusal.
