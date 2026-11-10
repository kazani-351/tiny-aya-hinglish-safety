# Hinglish safety on Tiny Aya: language drift, a small repair, and tool feedback

Entry for the Lexsi Labs AI Safety and Alignment Hackathon, Track 2a, problem statement "Multilingual and code-mixed safety".

Model: [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global). Tools chained: [SafeTune](https://github.com/Lexsi-Labs/SafeTune) (PlainSFT LoRA repair) and [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) (before/after scoring).

Write-up: [ARTICLE.md](https://github.com/kazani-351/tiny-aya-hinglish-safety/blob/main/ARTICLE.md). Adapter: [kazani351/tiny-aya-global-hinglish-lora](https://huggingface.co/kazani351/tiny-aya-global-hinglish-lora) (collection: [tiny-aya-hinglish-safety-model](https://huggingface.co/collections/kazani351/tiny-aya-hinglish-safety-model)). Labels and scores: [kazani351/tiny-aya-hinglish-safety-data](https://huggingface.co/datasets/kazani351/tiny-aya-hinglish-safety-data) (collection: [tiny-aya-hinglish-safety-data](https://huggingface.co/collections/kazani351/tiny-aya-hinglish-safety-data)).

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
| `reproduce_scores.ipynb` | CPU notebook: checks the published adapter and training rows at their pinned commits and runs `scorers.py` on them |
| `requirements-cpu.txt`, `requirements.txt` | Packages for the notebook (CPU) and for the GPU probes (Linux + NVIDIA) |
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

### 2. Check the published artifacts (CPU notebook, no login, ~2 GB download)

`reproduce_scores.ipynb` downloads the adapter and the 42 training rows at their pinned Hugging Face commits, checks the adapter's size and base model, and runs `scorers.py` on the training answers. It does not run the model. Needs Python 3.11 or newer (tested with 3.12 and 3.15). From the repo root:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-cpu.txt
.venv/bin/jupyter nbconvert --to notebook --execute reproduce_scores.ipynb --output reproduce_scores.out.ipynb
```

With uv instead: `uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -r requirements-cpu.txt`, then the same `jupyter` line. Or open the notebook in Jupyter or Google Colab and run all cells (on Colab its first cell clones this repo and installs `requirements-cpu.txt`). `fasttext` may compile from source (about a minute, needs a C++ compiler). The first run downloads GlotLID (1.7 GB) and takes several minutes; later runs take seconds. Expected output: `tensors=504  params=30,228,480`, then `lang_kept 42/42   script_kept 42/42`.

### 3. The GPU probes

**These require an NVIDIA GPU with CUDA.** The scripts call `.to("cuda")` and have no CPU or Mac fallback. A free Colab T4 is enough.

The probes were written for a Google Colab VM and launched with `run_gpu.sh` / `run_cpu.sh`, which drive the `colab` command-line client. To rerun one elsewhere:

- Environment: run them in a throwaway Colab runtime or virtual environment. Most probes `pip install` their own packages (`safetune`, `auditkit`, `fasttext`) into whatever Python runs them, and several first run `pip uninstall -y torchao`.
- Packages: `pip install -r requirements.txt` (Linux only). It pins torch, transformers, peft and accelerate to the versions that loaded the base model + published adapter on 2026-11-09, plus the pins inside the scripts (`safetune==0.1.10`, `auditkit==1.1.3`, `fasttext==0.9.3`). The original runs used the Colab versions of 2026-11-01 to 11-03; the adapter records peft 0.21.2.
- Paths: every probe reads inputs from and writes outputs to `/content/...`. Change those paths or create `/content/out`.
- Token: probes log in from `/content/hf.env` (a line `HF_TOKEN=...`). **Each probe deletes `/content/hf.env` after reading it**, so it does not stay on a shared VM: keep your own copy and recreate it before every run. The base model is gated: your Hugging Face account must accept Cohere's terms on its model page first (approval is automatic).
- GlotLID: pinned to revision `85cd6716494360367b75f642b5bc78667605d0b4` (the GlotLID v3 `model.bin`). On 2026-11-05 the repo's default `model.bin` became GlotLID v4, which labels some of our Hinglish replies as English; every number here uses the pinned version.
- Hardware: Tiny Aya Global (3.35B) fits a free T4 for inference and LoRA training. `probes/deeprefusal_fit_l4.py` ran on an HF Jobs L4.

`probes/full_rerun_probe.py` trains the PlainSFT adapter (2 epochs, lr 1e-4, LoRA r=16) and generates the probe 14 replies. Its one change to stock SafeTune is disclosed in the script header: `_save_merged` is patched to save the LoRA adapter instead of a merged 7 GB model.

The scripts read the training rows as `data/train_hinglish.jsonl`. The HF dataset ships them as `train_hinglish.csv`, because the dataset viewer needs one file format. To rebuild the JSONL from it (byte-identical to the file used in training; CSV turns an empty `style` into an empty cell, and this writes it back as ""), run this from the repo root:

```sh
mkdir -p data
curl -L -o train_hinglish.csv https://huggingface.co/datasets/kazani351/tiny-aya-hinglish-safety-data/resolve/ca83fcea77aa4a70e674c12425f56b05d28c64cb/train_hinglish.csv
python3 - <<'EOF'
import csv, json
with open("data/train_hinglish.jsonl", "w", encoding="utf-8") as f:
    for r in csv.DictReader(open("train_hinglish.csv", encoding="utf-8")):
        f.write(json.dumps({k: r[k] or "" for k in ("id", "type", "style", "prompt", "answer")}, ensure_ascii=False) + "\n")
EOF
```

`probes/full_rerun_probe.py` reads one input file, `rerun_inputs.jsonl`: the 42 training rows (`kind: "train"`, `id`, `prompt`, `answer`), the 16 over-refusal prompts (`kind: "pseudo"`, `id`, `prompt`), and the 16 translated harmful prompts (`kind: "harmful"`). The first two can be rebuilt from the HF dataset; the harmful prompts are withheld (see below), so the full rerun can't be repeated from public files alone: the script stops at a check that all 16 are present.

### What is not in this repo, and why

- **Model replies to harmful prompts** (`outs/*.jsonl`, `outs/auditkit_*.json`). Withheld because the prompts are harmful requests. Scripts that read them (`analyze_rerun.py`, `probes/auditkit_beforeafter.py`, `probes/regex_hits_check.py`, `probes/refusal_pattern_check.py`) cannot run from this repo alone; their outputs are in `results/`.
- **Translated harmful prompts** (`data/prompts.jsonl`). Withheld for the same reason. The English sources are public HarmBench prompts; the ids are in `selected_ids.json`.
- **What is in the probe scripts instead:** 37 short harmful requests: 17 in Hinglish and 12 in Hindi (Devanagari), written for this project (for example a fake ID, a stolen UPI PIN, a threatening letter), and 8 in English. Of the English ones, 4 are copied from SafeTune's README calibration pairs (including a pipe bomb and ransomware request) and 4 are behaviour descriptions for the HarmBench judge. They are placeholder refusal-training rows, steering calibration text and smoke-test prompts, and they appear with refusals only, never with completions. The scripts also quote one model reply, a Zulu refusal. None of this is the translated HarmBench evaluation set above.
- **Training rows** (`data/train_hinglish.jsonl`, 42 rows; `train_hinglish.csv` on HF, same fields) and the 16 pseudo-harmful prompts (`data/pseudo_harmful_hinglish.csv`) are not in this repo. They are published in the HF dataset collection ([kazani351/tiny-aya-hinglish-safety-data](https://huggingface.co/datasets/kazani351/tiny-aya-hinglish-safety-data)) under CC BY-NC 4.0 for research use. The 18 refusal rows pair a harmful Hinglish request with a refusal only, never with a completion.

## Timing

The rubric and decision rules were committed before the results they govern; the label files were committed before the key was opened. Git dates are self-reported, so two commit SHAs are anchored in Bitcoin with OpenTimestamps: `24ec288` and `6f19fb2` (`proofs/commit-<sha>.sha` holds the SHA, `.sha.ots` is its proof). Both are in block 974660 (2026-11-05 11:10 UTC). A commit SHA covers every commit before it, so this anchors the history up to `6f19fb2`, including the label commit `b5910a9`.

What this proves: those commits existed by 2026-11-05 11:10 UTC. What it does not prove: the order of commits before that, or the dates written inside them. Commits after `6f19fb2` are not stamped.

To check it:

1. The SHA is in this repo's history: `git merge-base --is-ancestor $(cat proofs/commit-6f19fb2e8ef4ae24a0846b832ca0f1451b849f44.sha) HEAD && echo ok`
2. The proof is anchored in Bitcoin: `pip install opentimestamps-client==0.8.0`, then `ots verify proofs/commit-<sha>.sha.ots`. That command needs a local Bitcoin node. Without one, run (it needs the same `opentimestamps-client` package) `python proofs/ots_roots.py proofs/commit-<sha>.sha.ots proofs/commit-<sha>.sha` and compare each printed merkle root with that block's merkle root on a block explorer. Each proof prints several blocks (one per timestamp calendar); the earliest, 974660, is the one that dates the commit.

## Licences

- Tiny Aya Global: CC BY-NC 4.0 with Cohere Labs' Acceptable Use Addendum. The adapter is shared under the same terms, noncommercial, research only.
- SafeTune and AuditKIT: Lexsi Labs Source Available License (LSAL) v1.2.
- HarmBench: MIT (github.com/centerforaisafety/HarmBench, checked 2026-11-08). GlotLID (cis-lmu/glotlid): Apache 2.0 plus notices, per its model card (checked 2026-11-08).

## AI assistance

Claude (Anthropic) wrote most of the code, drafted the training rows, translations and documents, and served as the second rater. kazani edited and fact-checked them, labelled the replies, and made every decision.
