# Probe 16: before/after through AuditKIT (2026-11-03) — the Track 2 chain SafeTune -> AuditKIT
ak.evaluate(model="precomputed") on probe 14's saved greedy replies (32 per condition: 16 harmful + 16 pseudo), scorers =
our scorers.py inside AuditKIT's scorer API (lang_kept: Hindi-family label, any script; script_kept: Latin script) +
AuditKIT's own RefusalDetector (refusal_en_regex). Then ak.compare(base_plain, X). AuditKIT 1.1.3, no network calls.
Per-condition result files (outs/auditkit_*.json) contain reply text: never publish them.

| Condition | lang_kept | script_kept | refusal_en_regex | ak.compare vs base: newly correct / newly wrong |
|---|---|---|---|---|
| base_plain | 0.72 | 0.72 | 0.03 | - |
| base_sys | 0.78 | 0.94 | 0.13 | 5 / 3 |
| base_fewshot | 0.94 | 1.00 | 0.00 | 9 / 2 |
| plainsft | 0.88 | 1.00 | 0.00 | 9 / 4 |

- **AuditKIT reproduces the repair:** both repairs move script_kept 0.72 -> 1.00 and lang_kept up by 0.16-0.22; ak.compare grades both PASS on these metrics.
- **Metric note:** AuditKIT's lang_kept (0.72 base) is looser than our headline "Hinglish kept" (Hindi-family AND Latin script, 14/32 = 0.44 base): a Devanagari reply counts as kept for lang_kept, not for script_kept. Report both; the strict one is the headline.
- **Tool-feedback finding:** AuditKIT's built-in RefusalDetector finds 0-4 refusals in 32 Hinglish replies per condition, where human raters find 4-7 refusals on 8 harmful prompts per condition. Worse, `ak.compare` grades the repairs **WARN** on refusal_en_regex ("-100%"), i.e. it reports a safety regression that the human labels show did not happen; the detector just can't read Hinglish refusals. This is GitHub issue material for AuditKIT (done-list item 8).
