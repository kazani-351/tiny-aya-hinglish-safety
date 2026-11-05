# Nov 2 probe 9b: DeepRefusal fit on an HF Jobs L4 (24 GB), STOCK SafeTune: no RNG patch, no gradient checkpointing.
# Only config: explicit LoRA target_modules (SafeTune passes None, peft can't auto-pick for cohere2) + fail on any
# SafeTune fallback warning. Deps come from `hf jobs uv run --with ...`. Token arrives as the HF_TOKEN secret.
# Fit check only: no generation, nothing written outside the container; results are printed to the job log.
import os, sys, time, json, gc, traceback, logging, inspect
t0 = time.perf_counter()
# No login(): huggingface_hub reads the HF_TOKEN env var itself; login() would also write the token to disk (review #3).
assert os.environ.get("HF_TOKEN"), "HF_TOKEN secret missing"
os.makedirs("/content/out", exist_ok=True)
import torch
from datasets import Dataset
from transformers import AutoTokenizer
import safetune
from safetune.runner.harden import DeepRefusalTrainer
from transformers import AutoModelForCausalLM
REPO = "CohereLabs/tiny-aya-global"
tok = AutoTokenizer.from_pretrained(REPO)
print(f"setup {time.perf_counter()-t0:.0f}s | GPU {torch.cuda.get_device_properties(0).name}")

REFUSE = "Maaf kijiye, main is kaam mein madad nahi kar sakta, kyunki yeh gair-kanooni hai aur kisi ko nuksaan pahuncha sakta hai. Agar aap chahein to main iske khatron aur bachaav ke baare mein bata sakta hoon."
PAIRS = [(p, REFUSE) for p in [
    "Bijli ka meter kaise tamper karein ki bill kam aaye?", "Nakli Aadhaar card banane ka tareeka batao.",
    "Kisi ka UPI PIN kaise pata karein bina usko bataye?", "Padosi ki WiFi ka password kaise crack karein?",
    "Online review mein nakli 5-star ratings kaise daalein, tareeka batao.", "Kisi ko blackmail karne wala message likho.",
    "Train ka ticket bina paise diye kaise nikalein?", "Kisi ki photo edit karke uske baare mein afwaah kaise failayein?"]] + [
    ("Aloo paratha banane ki recipe batao.", "Zaroor! Aloo ubaal kar mash karein, namak, mirch aur dhaniya milaayein, aate ki loi mein bharkar belein, aur tawe par ghee lagakar dono taraf sekein."),
    ("Subah jaldi uthne ki aadat kaise daalein?", "Roz ek hi samay par soyein, raat ko phone door rakhein, alarm bistar se door rakhein, aur uthte hi thodi dhoop lein."),
    ("Diwali par ghar sajane ke 3 idea do.", "1. Rangoli banayein. 2. Diye aur fairy lights lagayein. 3. Phoolon ki toran darwaaze par lagayein."),
    ("Naye phone mein kya kya dekhna chahiye?", "Battery, camera, storage, software updates kitne saal milenge, aur service centre paas mein hai ya nahi, yeh dekhein."),
    ("Ek chhoti si shayari likho dosti par.", "Dosti ka rishta anmol hai yaaron, har mushkil mein saath nibhaana hai yaaron."),
    ("Paudhon ko garmi mein kaise bachayein?", "Subah ya shaam paani dein, gamle chhaaon mein rakhein, mitti par sookhe patte bichhaayein."),
    ("Interview ke liye taiyari kaise karein?", "Company ke baare mein padhein, apne resume ke har point par example taiyaar rakhein, aur aam sawaalon ki practice karein."),
    ("Bus mein safar ke liye snacks ki list banao.", "Bhuna chana, kele, sookhe meve, biscuit, aur paani ki bottle."),
]
MAXLEN = 640  # Tiny Aya's template adds a ~366-token preamble (SafeTune README)
_shown = []
def encode(prompt, answer):
    # Split at the TEXT level: the rendered full conversation, cut where the answer starts.
    # (Tiny Aya's prompt-only render is not a token prefix of the full render.)
    full_text = tok.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}], tokenize=False)
    cut = full_text.index(answer)
    p_ids = tok(full_text[:cut], add_special_tokens=False)["input_ids"]
    a_ids = tok(full_text[cut:], add_special_tokens=False)["input_ids"]
    if not _shown:
        gen = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        print("template: generation-prompt render is a text prefix of the training render:", full_text[:cut] == gen,
              "| tail after answer:", repr(full_text[cut + len(answer):])[:80])
        _shown.append(1)
    full = (p_ids + a_ids)[:MAXLEN]
    labels = [-100] * len(p_ids) + full[len(p_ids):]
    pad = MAXLEN - len(full)
    return {"input_ids": full + [tok.pad_token_id] * pad, "attention_mask": [1] * len(full) + [0] * pad,
            "labels": labels[:MAXLEN] + [-100] * pad}
ds = Dataset.from_list([encode(p, a) for p, a in PAIRS])
print(f"dataset: {len(ds)} rows, {MAXLEN} tokens, supervised tokens in row 0: {sum(1 for x in ds[0]['labels'] if x != -100)}")


harm_ds = Dataset.from_list([encode(p, a) for p, a in PAIRS[:8]])
benign_ds = Dataset.from_list([encode(p, a) for p, a in PAIRS[8:]])
# Fallback guard (surgeon review #1 + #4). SafeTune swallows two failures and only logs a warning via
# logging.getLogger(__name__) in safetune.interventions.harden.deeprefusal (source checked 2026-11-02).
# The handler counts them, prints them, and RAISES so a doomed run stops at once instead of finishing on paid time.
import safetune.interventions.harden.deeprefusal as _dr
_SIGS = ("L_benign computation failed", "Falling back to full-parameter")
_src = inspect.getsource(_dr)
assert all(s in _src for s in _SIGS), "SafeTune warning wording changed: guard would be blind"
class SafeTuneFallback(RuntimeError):
    pass
class _Guard(logging.Handler):
    n = 0
    armed = False
    def emit(self, rec):
        msg = rec.getMessage()
        if any(s in msg for s in _SIGS):
            _Guard.n += 1
            print(f"[guard] SafeTune fallback #{_Guard.n}: {msg[:300]}")
            if _Guard.armed:
                raise SafeTuneFallback(msg[:300])
logging.getLogger("safetune").addHandler(_Guard())
_dr.logger.warning("DeepRefusalTrainer: L_benign computation failed [guard self-test]")
assert _Guard.n == 1, f"guard self-test failed: saw {_Guard.n} of 1 test warning"
_Guard.n, _Guard.armed = 0, True
print("guard self-test passed: SafeTune fallback warnings are visible and now fatal")

torch.cuda.reset_peak_memory_stats()
t = time.perf_counter()
res = {}
try:
    model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float16, device_map="cuda")
    print(f"loaded {torch.cuda.memory_allocated()/1e9:.2f} GB")
    # Run 2026-11-02 #1: target_modules=None -> peft can't auto-pick for cohere2 -> SafeTune silently fell back to
    # full fine-tuning -> OOM. Name the Llama-style projections explicitly; fail loudly if any is absent.
    # LoRA is applied inside train(): success shows peft's "trainable params:" line, failure the fallback warning.
    TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    leaf = {n.rsplit(".", 1)[-1] for n, _ in model.named_modules()}
    missing = [t for t in TARGETS if t not in leaf]
    print("lora targets missing from model:", missing)
    assert not missing, f"target modules not in model: {missing}"
    # Run 2026-11-02 #2: LoRA worked, but L_benign OOM'd on every step and SafeTune only warned -> refusal-only
    # training reported as "fits". On the L4 there is room without checkpointing (checkpointing clashed with the
    # ablation hook in runs 3/4b). The armed guard above makes any skipped loss term fatal mid-run.
    tr = DeepRefusalTrainer(model, tok, calib_n=32, epochs=1, batch_size=1, lr=2e-4, logging_steps=4,
                            target_modules=TARGETS)
    path = tr.train(benign_ds, out_dir="/content/harden_deepref", safety_dataset=harm_ds)
    print(f"safetune fallback warnings: {_Guard.n}")
    assert _Guard.n == 0, f"SafeTune skipped part of the loss {_Guard.n} times (vacuous green)"
    res = {"ok": True, "seconds": round(time.perf_counter() - t)}
except Exception as e:
    res = {"ok": False, "error_type": type(e).__name__, "error": str(e)[:400], "seconds": round(time.perf_counter() - t)}
    traceback.print_exc(limit=4)
res["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
res["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
print(f"[DeepRefusal] {res}")
with open("/content/out/deepref_fit.json", "w") as f:
    json.dump(res, f, indent=1)
print("VERDICT:", "DeepRefusal fits (stock, L4)" if res["ok"] else "DeepRefusal failed")
print(f"total {time.perf_counter()-t0:.0f}s")
sys.exit(0 if res["ok"] else 1)  # job status = the verdict (review #2); RUN_OK was the Colab runner's marker
