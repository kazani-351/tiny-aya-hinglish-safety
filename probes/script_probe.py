# Nov 1 probe 7: is Hinglish language loss a greedy artifact, and is it about SCRIPT or CODE-MIXING?
# Run: OUT=/content/out UPLOAD=data/prompts.jsonl EXEC_TIMEOUT=2400 ./run_gpu.sh probes/script_probe.py lexsi-script
# Unsteered Tiny Aya on the 16 harmful prompts in 3 forms: hi (Devanagari), hi_roman (same words, Latin script),
# hinglish (code-mixed, Latin). Decodes: greedy + 3 sampling seeds (temperature 0.7). GlotLID top label per reply.
# hi vs hi_roman isolates script; hi_roman vs hinglish isolates code-mixing. Replies -> /content/out only.
import subprocess, sys, os, time, json
t0 = time.perf_counter()
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fasttext==0.9.3"], capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-400:]
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
import torch, fasttext
from transformers import AutoTokenizer, AutoModelForCausalLM
lid = fasttext.load_model(hf_hub_download("cis-lmu/glotlid", "model.bin"))
HINDUSTANI = {"hin", "hif", "hns", "urd", "anp", "bho", "mai", "awa", "mag", "hne", "bjj"}
def langid(t):
    t = " ".join(t.split())
    return lid.f.predict(t, 1, 0.0, "strict")[0][1].replace("__label__", "") if t else "empty"
REPO = "CohereLabs/tiny-aya-global"
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float16, device_map="cuda").eval()
prompts = [json.loads(l) for l in open("/content/prompts.jsonl")]
DECODES = {"greedy": None, "s1": 1, "s2": 2, "s3": 3}
out = open("/content/out/script_probe.jsonl", "w")
res = {}
for form in ("hi", "hi_roman", "hinglish"):
    for dname, seed in DECODES.items():
        kept, labs = 0, []
        for p in prompts:
            inp = tok.apply_chat_template([{"role": "user", "content": p[form]}], add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True).to("cuda")
            if seed is None:
                o = model.generate(**inp, max_new_tokens=96, do_sample=False)
            else:
                torch.manual_seed(seed)
                o = model.generate(**inp, max_new_tokens=96, do_sample=True, temperature=0.7, top_p=1.0, top_k=0)
            reply = tok.decode(o[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
            l = langid(reply); k = int(l.split("_")[0] in HINDUSTANI); kept += k
            if not k: labs.append(l)
            out.write(json.dumps({"form": form, "decode": dname, "hb_id": p["hb_id"], "lid": l, "lang_kept": k, "reply": reply}, ensure_ascii=False) + "\n"); out.flush()
        res[(form, dname)] = kept
        print(f"{form:<9} {dname:<6} lang_kept {kept:>2}/16  lost-to: {sorted(labs)}")
out.close()
print("\nsummary (lang_kept/16):   greedy  s1  s2  s3")
for form in ("hi", "hi_roman", "hinglish"):
    print(f"  {form:<9}              " + "  ".join(f"{res[(form, d)]:>3}" for d in DECODES))
assert len(res) == 12
print(f"total {time.perf_counter()-t0:.0f}s")
print("RUN_OK")
