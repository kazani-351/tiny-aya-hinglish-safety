# Tiny Aya fit check on a free T4.
# Two ways to run it:
#   1. From Claude Code / terminal: ./run_gpu.sh fit_check.py   (token comes from ~/.config/hf.env)
#   2. Browser Colab: paste into one cell of a NEW notebook, T4 runtime, Secrets: HF_TOKEN on.
# Prompts are benign on purpose: this checks fit, speed and language quality, not safety.
import time, torch
from huggingface_hub import login, ModelCard
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

def _hf_login(path="/content/hf.env"):
    # CLI runs: token file uploaded by run_gpu.sh, read then deleted. Browser runs: Colab Secrets.
    import os
    if os.path.exists(path):
        token = None
        with open(path) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip()
        os.remove(path)
        if not token:
            raise SystemExit("FAIL: no HF_TOKEN= line in the uploaded token file")
    else:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
    login(token=token)
    from huggingface_hub import whoami
    me = whoami(token=token)
    role = ((me.get("auth") or {}).get("accessToken") or {}).get("role")
    print(f"HF login ok: user={me.get('name')} token_role={role}")

_hf_login()
REPO = "CohereLabs/tiny-aya-global"

assert torch.cuda.is_available(), "No GPU. Switch the runtime to T4 and run again."
gpu = torch.cuda.get_device_properties(0)
print(f"GPU: {gpu.name}, {gpu.total_memory/1e9:.1f} GB | transformers {transformers.__version__}")

langs = ModelCard.load(REPO).data.language or []
south_asian = [l for l in langs if l in {"hi", "mr", "bn", "gu", "pa", "ta", "te", "ur", "ne"}]
print(f"Model card lists {len(langs)} languages. Hindi listed: {'hi' in langs}. South Asian: {south_asian}")

t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(REPO)
# float16 because T4 has no native bfloat16. If replies come out as gibberish or NaN, change to torch.bfloat16.
model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float16, device_map="cuda")
print(f"Load: {time.perf_counter()-t0:.0f}s | GPU memory after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

def ask(prompt, max_new_tokens=128):
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to("cuda")
    plen = inputs["input_ids"].shape[1]
    torch.cuda.synchronize(); t = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize(); dt = time.perf_counter() - t
    n = out.shape[1] - plen
    return tok.decode(out[0, plen:], skip_special_tokens=True), dt, n

tests = {
    "Hindi": "मानसून में घर को सीलन से बचाने के तीन आसान तरीके बताइए।",
    "Hinglish": "Monsoon mein ghar ko seelan se bachane ke 3 easy tareeke batao.",
}
rates = []
for name, prompt in tests.items():
    text, dt, n = ask(prompt)
    rates.append(n / dt)
    print(f"\n[{name}] {n} tokens in {dt:.1f}s ({n/dt:.1f} tok/s)\n{text[:500]}")

peak = torch.cuda.max_memory_allocated() / 1e9
rate = min(rates)
print(f"\nPeak GPU memory: {peak:.2f} GB of {gpu.total_memory/1e9:.1f} GB")
print(f"Rough time for 144 replies x 256 tokens, one at a time: ~{144*256/rate/60:.0f} min")
print("VERDICT:", "FITS" if peak < 14 and rate >= 5 else "CHECK KILL CRITERIA IN PLAN.md")

print("RUN_OK")
