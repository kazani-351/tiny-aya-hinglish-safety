# Hinglish safety on Tiny Aya: language drift, a small repair, and tool feedback

Entry for the Lexsi Labs AI Safety and Alignment Hackathon, Track 2a, problem statement "Multilingual and code-mixed safety".

Model: [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global). Tools chained: [SafeTune](https://github.com/Lexsi-Labs/SafeTune) (PlainSFT LoRA repair) and [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) (before/after scoring).

Write-up: [K: article link]. Adapter: [K: HF model collection]. Labels and scores: [K: HF dataset collection].

## What is in this repo

| Path | What it is |
|---|---|
| `RUBRIC.md`, `DECISION_RULES_probe4.md` | Labelling rubric and decision rules, each committed before the results they govern (see git history) |
| `labels/` | Human labels (kazani) and blind LLM second-rater labels (Claude). No reply text |
| `outs/rerun_label_key.csv`, `outs/label_key.csv` | Unblinding keys: row to condition, prompt id and GlotLID label. No reply text |
| `analyze_labels.py` | Rebuilds the probe 14 label tables and agreement figures from the files above |
| `results/` | One markdown file per probe or check, with the numbers the write-up cites |
| `probes/` | The scripts that ran on GPU (Colab T4, one HF Jobs L4 run) |
| `scorers.py` | Language-kept and script-kept scorers (GlotLID), used directly and inside AuditKIT |
| `drafts/` | Model card and GitHub issue drafts |
| `proofs/` | OpenTimestamps proofs for commit SHAs (see "Timing" below) |

## Reproduce

### 1. Label tables and agreement (CPU, seconds, no downloads)

```sh
python3 analyze_labels.py
```

Needs Python 3.9+ and nothing else. Expected output includes:

- Hinglish kept per condition, strict, kazani: base 9/16, system prompt 9/16, few-shot 16/16, PlainSFT adapter 14/16
- Cohen's kappa, kazani vs Claude v2, Hinglish strict: 0.87 (observed 0.953, chance 0.633)

### 2. The GPU probes

The probes were written for a Google Colab VM and launched with `run_gpu.sh` / `run_cpu.sh`, which drive the `colab` command-line client. To rerun one elsewhere:

- Paths: every probe reads inputs from and writes outputs to `/content/...`. Change those paths or create `/content/out`.
- Token: probes log in from `/content/hf.env` (a line `HF_TOKEN=...`). The base model is gated: your Hugging Face account must accept Cohere's terms first.
- Packages: pinned inside each script (`safetune==0.1.10`, `auditkit==1.1.3`, `fasttext==0.9.3`). torch, transformers and peft were the Colab versions of 2026-11-01 to 11-03; the adapter records peft 0.21.2.
- Hardware: Tiny Aya Global (3.35B) fits a free T4 for inference and LoRA training. `probes/deeprefusal_fit_l4.py` ran on an HF Jobs L4.

`probes/full_rerun_probe.py` trains the PlainSFT adapter (2 epochs, lr 1e-4, LoRA r=16) and generates the probe 14 replies. Its one change to stock SafeTune is disclosed in the script header: `_save_merged` is patched to save the LoRA adapter instead of a merged 7 GB model.

### What is not in this repo, and why

- **Model replies to harmful prompts** (`outs/*.jsonl`, `outs/auditkit_*.json`). Withheld because the prompts are harmful requests. Scripts that read them (`analyze_rerun.py`, `probes/auditkit_beforeafter.py`, `probes/regex_hits_check.py`, `probes/refusal_pattern_check.py`) cannot run from this repo alone; their outputs are in `results/`.
- **Translated harmful prompts** (`data/prompts.jsonl`). Withheld for the same reason. The English sources are public HarmBench prompts; the ids are in `selected_ids.json`.
- **What is in the probe scripts instead:** 37 short harmful requests: 17 in Hinglish and 12 in Hindi (Devanagari), written for this project (for example a fake ID, a stolen UPI PIN, a threatening letter), and 8 in English. Of the English ones, 4 are copied from SafeTune's README calibration pairs (including a pipe bomb and ransomware request) and 4 are behaviour descriptions for the HarmBench judge. They are placeholder refusal-training rows, steering calibration text and smoke-test prompts, and they appear with refusals only, never with completions. The scripts also quote one model reply, a Zulu refusal. None of this is the translated HarmBench evaluation set above.
- **Training rows** (`data/train_hinglish.jsonl`) and the pseudo-harmful prompts: [K: decide; published on HF or withheld]. Note: 9 prompts and 5 answers from the 12 benign training rows reuse placeholder text that is already in the probe scripts, so those benign rows are public either way. No refusal or harmful-sounding training row appears in this repo.

## Timing

The rubric and decision rules were committed before the results they govern; the label files were committed before the key was opened. Git dates are self-reported, so two commit SHAs are anchored in Bitcoin with OpenTimestamps: `24ec288` and `6f19fb2` (`proofs/commit-<sha>.sha` holds the SHA, `.sha.ots` is its proof). Both are in block 974660 (2026-11-05 11:10 UTC). A commit SHA covers every commit before it, so this anchors the history up to `6f19fb2`, including the label commit `b5910a9`.

What this proves: those commits existed by 2026-11-05 11:10 UTC. What it does not prove: the order of commits before that, or the dates written inside them. Commits after `6f19fb2` are not stamped.

To check it:

1. The SHA is in this repo's history: `git merge-base --is-ancestor $(cat proofs/commit-6f19fb2e8ef4ae24a0846b832ca0f1451b849f44.sha) HEAD && echo ok`
2. The proof is anchored in Bitcoin: `pip install opentimestamps-client==0.8.0`, then `ots verify proofs/commit-<sha>.sha.ots`. That command needs a local Bitcoin node. Without one, run `python proofs/ots_roots.py proofs/commit-<sha>.sha.ots proofs/commit-<sha>.sha` and compare each printed merkle root with that block's merkle root on a block explorer.

## Licences

- Tiny Aya Global: CC BY-NC 4.0 with Cohere Labs' Acceptable Use Addendum. The adapter is shared under the same terms, noncommercial, research only.
- SafeTune and AuditKIT: Lexsi Labs Source Available License (LSAL) v1.2.
- HarmBench: [K: verify licence]. GlotLID: [K: verify licence].

## AI assistance

Claude (Anthropic) wrote most of the code, drafted the training rows, translations and documents, and served as the second rater. kazani edited and fact-checked them, labelled the replies, and made every decision.
