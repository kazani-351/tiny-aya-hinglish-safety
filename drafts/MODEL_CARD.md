---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
library_name: peft
pipeline_tag: text-generation
language:
- hi
- en
tags:
- base_model:adapter:CohereLabs/tiny-aya-global
- lora
- peft
- safetune
- hinglish
- code-mixing
- multilingual-safety
- research-only
---

# Tiny Aya Global: Hinglish language-keeping LoRA (research only)

This is a LoRA adapter for [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global). It was trained to fix one problem: on Hinglish prompts (Hindi and English mixed, written in Latin script), the base model often replies in another language. It switches to English, to Hindi in Devanagari, or to an unrelated language.

The adapter was made for the Lexsi Labs hackathon, Track 2a #4 ("Multilingual and code-mixed safety"). It is a research artifact. **It is not for deployment.**

## What it does and does not change

Greedy decoding. Human labels cover 16 replies per condition (8 harmful prompts, 8 harmless prompts that sound harmful). The AuditKIT row covers all 32 replies per condition (16 + 16).

| | Base model | Base + system prompt | Base + 3 few-shot examples | This adapter |
|---|---|---|---|---|
| Replies in Hinglish (kazani's label, strict, of 16) | 9/16 | 9/16 | 16/16 | 14/16 |
| Harmful prompts: refused / complied / other (of 8) | 4 / 2 / 2 | 6 / 2 / 0 | 6 / 2 / 0 | 6 / 1 / 1 |
| Harmless prompts that sound harmful: over-refused (of 8) | 0 | 0 | 0 | 0 |
| Latin script kept, AuditKIT `script_kept` (of 32 replies) | 0.72 | 0.94 | 1.00 | 1.00 |

- **The adapter probably closes most of the language gap.** Hinglish replies go from 9/16 to 14/16 by kazani's strict count, which treats a reply mixing Hinglish and English sentences as not kept. On those labels alone the gain is not significant (paired exact McNemar, 6 gained, 1 lost, p = 0.125). The second rater and GlotLID both count 16/16 (p of 0.008 or lower), and a GlotLID check over 32 prompts clears the bar in 7 of 7 runs. Three few-shot examples do better on kazani's labels (16/16, p = 0.016). See results/significance_probe14.md.
- **No refusal change was detected.** 4 to 6 of 8 harmful prompts are refused in every condition. With 8 prompts per kind we do not report differences under about 3 rows (RUBRIC A4), so we can't claim a change either way, and a smaller real change could go undetected.
- **It does not cause over-refusal** on these 8 prompts.
- **Three few-shot examples do at least as well.** Head to head, few-shot (16/16) and this adapter (14/16) differ by less than 3 rows, so we can't rank them at this sample size; only few-shot's gain over the base is significant on kazani's labels. The adapter is useful when you can't control the prompt, not because it beats prompting.
- **Fluency is still weak.** Most replies in every condition were labelled "partly makes sense". Tiny Aya's Hinglish is fragile however it is prompted.
- **No copying.** 0 of 64 adapter replies (32 greedy, 32 sampled) repeat any of the 42 training answers (longest shared run 3 to 5 words, against 4 for the base model). 4 of the 64 use a stock refusal phrase that also appears in the training set.

### Who labelled
- **Rater 1:** kazani, blind to condition. By the repository's commit order, the labels were committed (b5910a9) before the condition key was opened. Commit order is self-reported. The history up to commit 6f19fb2, which includes b5910a9, is timestamped with OpenTimestamps in Bitcoin block 974660 (2026-11-05 11:10 UTC). That proves those commits existed by that time, not their order; see the README's Timing section.
- **Rater 2:** a blind LLM rater (Claude), run on a rubric that did not state the expected result (RUBRIC A6). Agreement with rater 1 on "is this Hinglish": Cohen's kappa 0.87 (chance agreement 0.63).
- A decoder check scored automatically with GlotLID (not human labels), over 7 runs on all 32 prompts (greedy plus 6 sampled), found both repairs keep Hinglish at least 3 replies above the base model in 7 of 7 runs.

Small n. Treat every number above as a pilot result, not a benchmark.

## How it was made

- **Base model:** CohereLabs/tiny-aya-global, unchanged weights.
- **Method:** SafeTune 0.1.10 `PlainSFTTrainer` (SafeTune's "no defense" fine-tuning baseline), used here as a targeted language-keeping repair. LoRA r=16, alpha=32, dropout 0.05, on q/k/v/o/gate/up/down projections. 2 epochs, learning rate 1e-4, batch size 1, max length 640, loss on the answer tokens only. Free Colab T4.
- **One change to stock SafeTune:** SafeTune saves a merged full model. Our script patched `_save_merged` to save only this adapter. The adapter was reloaded and checked against the in-memory model.
- **Training data:** 42 Hinglish rows: 18 refusals to harmful requests, 12 helpful answers to benign requests, 12 helpful answers to requests that sound harmful but are not. Claude drafted the rows. kazani edited them, fact-checked the helpful answers against official sources, and cut 6 refusal rows that were unfit for a public set (48 to 42). The 16 pseudo-harmful evaluation prompts were made the same way.
- **Evaluation prompts:** 16 HarmBench prompts translated to Hinglish (Claude draft, hand-fixed by kazani) and the 16 pseudo-harmful prompts above. The translated harmful prompts are not published.

## How to use (research only)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", torch_dtype="auto")
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
model = PeftModel.from_pretrained(base, "kazani351/tiny-aya-global-hinglish-lora")
```

You need access to the base model on Hugging Face.

## Limits and risks

- Tested on 32 prompts in one language mix. Nothing here says the adapter helps on other code-mixed languages or other prompt types.
- Refusal is unchanged, not improved. The base model complies with some harmful prompts, and so does this adapter (1 of 8 here, mostly misinformation prompts).
- Keeping the language is not the same as being safe or correct. A reply can stay in Hinglish and still be wrong or harmful.
- Not reviewed for production use. Do not deploy it.

## Licence and attribution

- **Base model:** [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) by Cohere Labs, released under [CC BY-NC 4.0](https://cohere.com/cohere-labs-cc-by-nc-license) with Cohere Labs' Acceptable Use Addendum. This adapter is modified from it as described in "How it was made" and is shared under the same licence, for noncommercial use only.
- **Acceptable use:** use of this adapter is subject to the [Cohere Labs Acceptable Use Policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy).
- **Training tool:** trained with [SafeTune](https://github.com/Lexsi-Labs/SafeTune), licensed under the Lexsi Labs Source Available License (LSAL) v1.2. Per LSAL, this checkpoint is for noncommercial research and must not be deployed in production.

## Credits

- Prior work this builds on: Kachireddy (2026), "A Measurement Crisis in Multilingual LLM Safety", SSRN [10.2139/ssrn.7152058](https://doi.org/10.2139/ssrn.7152058), on refusal checks that undercount Indic refusals; Krasnodębska, Kusa and Lipani (Findings of ACL 2026, [10.18653/v1/2026.findings-acl.1537](https://doi.org/10.18653/v1/2026.findings-acl.1537)), on measuring reply language next to refusal; Yadav (LoResLM 2026, [10.18653/v1/2026.loreslm-1.3](https://doi.org/10.18653/v1/2026.loreslm-1.3)), on Hinglish drift into English. This adapter tests one repair on one model; it does not claim those findings.
- Write-up: [ARTICLE.md](https://github.com/kazani-351/tiny-aya-hinglish-safety/blob/main/ARTICLE.md). Code and scorers: [GitHub](https://github.com/kazani-351/tiny-aya-hinglish-safety). Training rows, labels and per-reply scores: [dataset](https://huggingface.co/datasets/kazani351/tiny-aya-hinglish-safety-data).

## AI assistance

Claude (Anthropic) drafted the training rows, the prompt translations, this model card, and served as the second rater. kazani edited and fact-checked all of it and made every decision. Human labels are kazani's.
