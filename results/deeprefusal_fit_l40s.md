# DeepRefusal fit on an HF Jobs L40S (48 GB), stock SafeTune 0.1.10 (2026-11-05)
Job kazani351/6aec51c00e0338d1bb806797, flavor l40sx1, script `probes/deeprefusal_fit_l4.py` unchanged (the name says L4; only the hardware changed). Log: ~/.claude/logs/lexsi-deepref-l40s-2026-11-05.log.
Config: stock SafeTune DeepRefusalTrainer, no RNG patch, no gradient checkpointing; the only change is explicit LoRA target modules (q/k/v/o/gate/up/down_proj), because SafeTune passes `target_modules=None` (GitHub issue draft SafeTune #1). Fit check only: 16 placeholder Hinglish rows (8 refusal, 8 helpful), MAXLEN 640, batch 1, 1 epoch = 8 steps. SafeTune derived its refusal direction from its default calibration data (it downloaded a 330k + 30k harmful set and a 52k instruction set). No generation, no evaluation.

| Measure | Value |
|---|---|
| LoRA on | yes: 30,228,480 trainable of 3,379,456,000 (0.89%) |
| SafeTune fallback warnings (full fine-tune fallback, skipped L_benign) | **0**: the benign term ran on every step |
| Peak GPU memory | 31.29 GB allocated, 32.2 GB reserved |
| Training time | 43 s for 8 steps (loss 4.52 -> 2.99) |
| Job wall time | 146 s running (queued ~58 min for hardware first, not billed) |
| Cost | about $0.07 at the $1.80/h list price |

Reading:
- Stock DeepRefusal on Tiny Aya Global needs about 32 GB at MAXLEN 640, batch 1. That is why it OOM'd on the T4 (15 GB) and the L4 (22 GB, run 5) and fits on a 48 GB L40S with room to spare.
- This answers "does it fit", not "does it work": nothing here says DeepRefusal keeps Hinglish or changes refusal. A real run needs the 42-row training set, the probe 14 evaluation, and labels.
- Article section 8 ("with more credits"): a full DeepRefusal run on the real data would cost well under $1 of GPU time on an L40S; the binding cost is labelling time, not credits.
