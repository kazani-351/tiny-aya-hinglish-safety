# Nov 1 probe 9: does SafeTune DeepRefusalTrainer (a real safety method, one model copy) fit Tiny Aya on a free T4?
# Run: OUT=/content/out EXEC_TIMEOUT=3000 ./run_gpu.sh probes/deeprefusal_fit_probe.py lexsi-deepref
# Test A: PlainSFTTrainer (LoRA, no reference model) on 16 tiny Hinglish examples (8 refusals, 8 helpful), batch 1.
# Test B: SafeGradTrainer (loads a frozen reference copy too) on the same data: expected to be tight / OOM on 15 GB.
# Reports: success or the exact error type, peak GPU memory, seconds per step, and a language-ID check of 2 replies
# from the A-trained model on prompts NOT in the study set. Fit check only: the data here is a placeholder for CuratorKIT.
import subprocess, sys, os, time, json, gc, traceback
t0 = time.perf_counter()
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "safetune==0.1.10", "fasttext==0.9.3"], capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-500:]
from huggingface_hub import login, hf_hub_download
def _hf_login(path="/content/hf.env"):
    token = None
    with open(path) as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip()
    os.remove(path)
    assert token
    login(token=token)
_hf_login()
os.makedirs("/content/out", exist_ok=True)
import torch, fasttext
from datasets import Dataset
from transformers import AutoTokenizer
import safetune
from safetune.runner.harden import DeepRefusalTrainer
from transformers import AutoModelForCausalLM
lid = fasttext.load_model(hf_hub_download("cis-lmu/glotlid", "model.bin"))
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
torch.cuda.reset_peak_memory_stats()
t = time.perf_counter()
res = {}
try:
    model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float16, device_map="cuda")
    print(f"loaded {torch.cuda.memory_allocated()/1e9:.2f} GB")
    tr = DeepRefusalTrainer(model, tok, calib_n=32, epochs=1, batch_size=1, lr=2e-4, logging_steps=4)
    path = tr.train(benign_ds, out_dir="/content/harden_deepref", safety_dataset=harm_ds)
    res = {"ok": True, "seconds": round(time.perf_counter() - t), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    del model, tr; gc.collect(); torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map="cuda").eval()
    for p in ["Kisi ki Instagram ID par nakli followers kaise badhayein?", "Monsoon mein kapde jaldi kaise sukhayein?"]:
        inp = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
        o = m.generate(**inp, max_new_tokens=64, do_sample=False)
        reply = tok.decode(o[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        l = lid.f.predict(" ".join(reply.split()), 1, 0.0, "strict")[0][1].replace("__label__", "")
        print(f"  trained-model reply lid: {l}")
        with open("/content/out/deepref_fit_replies.jsonl", "a") as f:
            f.write(json.dumps({"prompt": p, "reply": reply, "lid": l}, ensure_ascii=False) + "\n")
except Exception as e:
    res = {"ok": False, "error_type": type(e).__name__, "error": str(e)[:400], "seconds": round(time.perf_counter() - t),
           "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    traceback.print_exc(limit=4)
print(f"[DeepRefusal] {res}")
with open("/content/out/deepref_fit.json", "w") as f:
    json.dump(res, f, indent=1)
print("VERDICT:", "DeepRefusal fits" if res["ok"] else "DeepRefusal failed")
print(f"total {time.perf_counter()-t0:.0f}s")
print("RUN_OK")
