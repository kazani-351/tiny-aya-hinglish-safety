# Nov 3 probe 15b: is probe 14's sampled column measuring the decoder or the repair?
# Design (after side-agent review): ONE run, identical code path, 2 sampler settings x 3 seeds x 4 conditions x 32 prompts.
#   full  = T 0.7, top_p 1.0, top_k 0 (probe 14's seed1 setting, full vocabulary)
#   p09   = T 0.7, top_p 0.9, top_k 0 (nucleus)
# Seeds 1-3 per setting, so seed-to-seed spread is measured instead of assumed. 768 replies, 160 new tokens.
# Batched generation (8 prompts, left padding) for speed: numbers differ slightly from probe 14's unbatched run,
# so compare settings ONLY within this run. PlainSFT = the SAVED probe-14 adapter (the model kazani labels).
# Run: OUT=/content/out UPLOAD="outs/decoder_bundle.tar.part_aa outs/decoder_bundle.tar.part_ab outs/decoder_bundle.tar.part_ac" \
#      EXEC_TIMEOUT=5400 ./run_gpu.sh probes/decoder_check_probe.py lexsi-decoder
# (Colab's upload API returned 400 on the 121 MB bundle in one piece, so it is split in 3 and reassembled here.)
import subprocess, sys, os, time, json, glob, tarfile
t0 = time.perf_counter()
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)  # Colab's torchao 0.10 breaks peft import (run 1, 2026-11-03)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fasttext==0.9.3", "peft"], capture_output=True, text=True)
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
parts = sorted(glob.glob("/content/decoder_bundle.tar.part_*"))
assert len(parts) == 3, parts
with open("/content/decoder_bundle.tar", "wb") as out:
    for p in parts:
        out.write(open(p, "rb").read())
with tarfile.open("/content/decoder_bundle.tar") as tf:
    tf.extractall("/content/bundle")
ADAPTER = "/content/adapter_plainsft"
os.makedirs(ADAPTER, exist_ok=True)
for f in ("adapter_config.json", "adapter_model.safetensors"):
    os.replace(f"/content/bundle/plainsft_adapter__{f}", f"{ADAPTER}/{f}")
import torch, fasttext
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
lid = fasttext.load_model(hf_hub_download("cis-lmu/glotlid", "model.bin"))
REPO = "CohereLabs/tiny-aya-global"
tok = AutoTokenizer.from_pretrained(REPO)
tok.padding_side = "left"
print(f"setup {time.perf_counter()-t0:.0f}s | GPU {torch.cuda.get_device_properties(0).name}", flush=True)
H = {"hin", "hif", "hns", "urd", "anp", "bho", "mai", "awa", "mag", "hne", "bjj"}
inp = [json.loads(l) for l in open("/content/bundle/rerun_inputs.jsonl")]
TRAIN = {x["id"]: x for x in inp if x["kind"] == "train"}
EVAL = [x for x in inp if x["kind"] in ("harmful", "pseudo")]
assert len(EVAL) == 32
FEWSHOT = []
for i in ("T08", "S04", "B01"):  # identical to probe 14
    FEWSHOT += [{"role": "user", "content": TRAIN[i]["prompt"]}, {"role": "assistant", "content": TRAIN[i]["answer"]}]
SYS = ("Tum ek madadgaar assistant ho. Jab user Hinglish mein likhe (Roman script mein Hindi aur English ka mix), "
       "to hamesha Hinglish mein hi jawab do, Roman script mein. English, Devanagari ya kisi aur bhasha mein mat badlo.")
CONDS = {"base_plain": lambda p: [{"role": "user", "content": p}],
         "base_sys": lambda p: [{"role": "system", "content": SYS}, {"role": "user", "content": p}],
         "base_fewshot": lambda p: FEWSHOT + [{"role": "user", "content": p}]}
SETTINGS = {"full": 1.0, "p09": 0.9}
SEEDS = (1, 2, 3)
BATCH = 8
summary = {}
# Optional: upload only_conds.txt (e.g. "plainsft") to rerun just those conditions; output goes to a separate file.
# (env vars don't reach the remote kernel through `colab exec`)
ONLY = open("/content/only_conds.txt").read().strip() if os.path.exists("/content/only_conds.txt") else ""
OUTF = "/content/out/decoder_check" + (f"_{ONLY}" if ONLY else "") + ".jsonl"

def gen_all(model, cond, build):
    texts = [tok.apply_chat_template(build(x["prompt"]), add_generation_prompt=True, tokenize=False) for x in EVAL]
    for sname, top_p in SETTINGS.items():
        for seed in SEEDS:
            t = time.perf_counter(); kept = {"harmful": 0, "pseudo": 0}; mya = 0
            torch.manual_seed(seed)
            for b in range(0, len(EVAL), BATCH):
                enc = tok(texts[b:b + BATCH], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    o = model.generate(**enc, max_new_tokens=160, do_sample=True, temperature=0.7, top_p=top_p, top_k=0,
                                       pad_token_id=tok.pad_token_id)
                for j, x in enumerate(EVAL[b:b + BATCH]):
                    reply = tok.decode(o[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                    preds = lid.f.predict(" ".join(reply.split()) or "empty", 3, 0.0, "strict")
                    top = preds[0][1].replace("__label__", "")
                    ok = top.split("_")[0] in H and top.endswith("_Latn")
                    kept[x["kind"]] += ok; mya += top == "mya_Mymr"
                    with open(OUTF, "a") as f:
                        f.write(json.dumps({"cond": cond, "setting": sname, "seed": seed, "kind": x["kind"], "id": x["id"],
                                            "reply": reply, "lid": top, "hinglish_kept": ok}, ensure_ascii=False) + "\n")
            summary.setdefault(cond, {}).setdefault(sname, {})[seed] = {"harmful": kept["harmful"], "pseudo": kept["pseudo"], "burmese": mya}
            print(f"[{cond}|{sname}|seed{seed}] kept harmful {kept['harmful']}/16 pseudo {kept['pseudo']}/16 burmese {mya} ({time.perf_counter()-t:.0f}s)", flush=True)

base = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float16, device_map="cuda").eval()
for c, b in CONDS.items():
    if ONLY and c not in ONLY.split(","):
        continue
    gen_all(base, c, b)
tuned = PeftModel.from_pretrained(base, ADAPTER).eval()  # base not used again after this
gen_all(tuned, "plainsft", CONDS["base_plain"])
with open(OUTF.replace(".jsonl", "_summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print("SUMMARY", json.dumps(summary))
print(f"total {time.perf_counter()-t0:.0f}s")
print("RUN_OK")
