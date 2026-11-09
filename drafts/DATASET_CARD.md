---
license: cc-by-nc-4.0
language:
- hi
- en
pretty_name: Hinglish language-keeping repair data and labels (Tiny Aya Global)
size_categories:
- n<1K
task_categories:
- text-generation
configs:
- config_name: traces_probe14
  default: true
  data_files:
  - split: train
    path: traces/probe14_rerun_scores.csv
- config_name: traces_probe15b
  data_files:
  - split: train
    path: traces/probe15b_decoder_scores.csv
- config_name: over_refusal_prompts
  data_files:
  - split: train
    path: pseudo_harmful_hinglish.csv
- config_name: labels_probe4_kazani
  data_files:
  - split: train
    path: labels/probe4_base_hinglish_kazani.csv
- config_name: labels_probe14_kazani
  data_files:
  - split: train
    path: labels/probe14_rerun_kazani.csv
- config_name: labels_probe14_rater2_v1
  data_files:
  - split: train
    path: labels/probe14_rerun_llm.csv
- config_name: labels_probe14_rater2_v2
  data_files:
  - split: train
    path: labels/probe14_rerun_llm_v2.csv
- config_name: label_key_probe4
  data_files:
  - split: train
    path: labels/label_key.csv
- config_name: label_key_probe14
  data_files:
  - split: train
    path: labels/rerun_label_key.csv
- config_name: train_rows
  data_files:
  - split: train
    path: train_hinglish.csv
tags:
- hinglish
- code-mixing
- multilingual-safety
- refusal
- research-only
---

# Hinglish language-keeping repair: training rows, over-refusal prompts and labels

This dataset holds the data and labels behind a small study of Tiny Aya Global ([CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global)) on harmful prompts written in Hinglish (Hindi and English mixed, in Latin script). The base model often leaves Hinglish on these prompts. The study trained a LoRA adapter to keep the reply in Hinglish and checked that refusal did not drop. It was made for the Lexsi Labs hackathon, Track 2a ("Multilingual and code-mixed safety"). It is for research only.

Write-up: [ARTICLE.md](https://github.com/kazani-351/tiny-aya-hinglish-safety/blob/main/ARTICLE.md). Adapter: [kazani351/tiny-aya-global-hinglish-lora](https://huggingface.co/kazani351/tiny-aya-global-hinglish-lora). Code and scorers: [GitHub](https://github.com/kazani-351/tiny-aya-hinglish-safety).

## Files

- **`train_hinglish.csv`** (42 rows): the adapter's training set. Fields: `id`, `type`, `style`, `prompt`, `answer`. Types: 18 `refuse` rows (a harmful request in Hinglish with a refusal in Hinglish), 12 benign rows with helpful answers, and 12 rows that sound harmful but are fine to answer, with helpful answers.
- **`pseudo_harmful_hinglish.csv`** (16 rows): the over-refusal test set. Harmless Hinglish prompts that sound harmful (for example how to check a phone for malware). Fields: `id`, `prompt` (the Hinglish prompt used), `english_gloss`, `why_it_is_fine_to_answer`, `sounds_like` (the harmful topic it resembles).
- **`labels/`**: per-reply labels for the evaluation runs (refusal label, whether the reply makes sense, reply language, unsure flag, short note). kazani's labels for the base model (probe 4, 16 rows) and the repair rerun (probe 14, 64 rows), plus a second rater's labels for probe 14 (Claude, blind to condition; v1 and a v2 re-run). The notes describe replies; they do not quote them.
- **Label keys** (`labels/label_key.csv`, `labels/rerun_label_key.csv`): map each blind row to its condition, so the before/after numbers can be recomputed from the labels.
- **`traces/`**: per-reply automatic scores, with no reply or prompt text. `probe14_rerun_scores.csv` (256 rows: 4 conditions x greedy and one sampled run x 16 harmful and 16 over-refusal prompts) and `probe15b_decoder_scores.csv` (768 rows: 4 conditions x 2 decoding settings x 3 seeds x 32 prompts). Columns: the run keys (`cond`, `decode` or `setting` and `seed`, `kind`, `id`), `glotlid_label` (top GlotLID label, e.g. `hif_Latn`), `hinglish_kept` (as scored in the study), `lang_kept` (label in the Hindustani family), `script_kept` (Latin script), `loop_flag` (`ok`, `loop`, `vacuous` or `empty`, from the repo's `flag_loops.py`; it undercounts bad replies). Harmful prompt ids are HarmBench ids (`hb13` etc.); over-refusal ids match `pseudo_harmful_hinglish.csv`. The greedy rows reproduce the article's numbers (for example `lang_kept` 0.719 for the base model, the same as AuditKIT).

## What is not here

- **Model replies to the harmful prompts.** Withheld.
- **The 16 translated HarmBench evaluation prompts.** Withheld. The English sources are public HarmBench prompts; their ids are in the GitHub repo (`selected_ids.json`).

## Content warning

The 18 `refuse` rows contain harmful requests (for example forging a degree certificate or reading someone's messages without consent). Each is paired only with a refusal, never with a completion. Fourteen state the refusal outright; four (T12, T14, T16, T18) decline by redirecting to a safe alternative without a refusal sentence. No answer names a method, tool or step for the harmful request. Six rows judged unfit for a public set were dropped before training.

Known issue in one row: T11 asks for a full home setup to grow and sell ganja. Its answer offers "an overview of applicable licensing rules", which suggests a licensed route exists. Under India's NDPS Act, 1985, cannabis cultivation is prohibited except under narrow government permission (Section 14: fibre, seed or horticulture; Section 10: state rules, within Section 8's medical or scientific limit). No licence covers a home grow-and-sell setup. The row is kept as trained so the adapter stays reproducible; don't reuse its answer as is.

## How it was made

The training rows and the over-refusal prompts were drafted by Claude (Anthropic), then edited by kazani. kazani cut the training set from 48 rows to 42 and checked the answers in the 12 rows that sound harmful against official sources. The second-rater labels are Claude's, made blind to condition. kazani's labels are kazani's own.

## Licence

CC BY-NC 4.0, noncommercial use only. Use for research on multilingual safety. Do not use the refusal rows to train a model to comply with the requests.
