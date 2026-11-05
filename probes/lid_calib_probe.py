# Nov 3 probe 13 (CPU): can GlotLID tell real Hinglish apart? Calibrates the "Hinglish kept" rule used in every
# language-kept number (probe 7 and probe 11): kept = top-1 label in the Hindustani cluster (scorers.py) [+ _Latn in probe 11].
# Run: OUT=/content/out UPLOAD=outs/lid_calib_input.jsonl ./run_cpu.sh probes/lid_calib_probe.py lexsi-lidcal
# Known-language texts: 16 prompts x {hinglish hand-fixed, hinglish draft, hi_roman, Devanagari hi, English} from
# data/prompts*.jsonl (written/fixed by kazani), plus 8 Hinglish reply-style sentences (the probe-11 helpful answers,
# Claude-written, so a weaker check). No model replies involved. Writes per-text labels to /content/out (gitignored).
import subprocess, sys, json, collections
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fasttext==0.9.3"], capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-500:]
import fasttext, os
if os.path.exists("/content/hf.env"):
    os.remove("/content/hf.env")  # the runner always uploads the token; this probe doesn't need it
from huggingface_hub import hf_hub_download  # GlotLID is ungated: no token needed
lid = fasttext.load_model(hf_hub_download("cis-lmu/glotlid", "model.bin"))
H = {"hin", "hif", "hns", "urd", "anp", "bho", "mai", "awa", "mag", "hne", "bjj"}
rows = [json.loads(l) for l in open("/content/lid_calib_input.jsonl")]
rows += [{"set": "hinglish_reply_style", "hb_id": i, "text": t} for i, t in enumerate([
    "Zaroor! Aloo ubaal kar mash karein, namak, mirch aur dhaniya milaayein, aate ki loi mein bharkar belein, aur tawe par ghee lagakar dono taraf sekein.",
    "Roz ek hi samay par soyein, raat ko phone door rakhein, alarm bistar se door rakhein, aur uthte hi thodi dhoop lein.",
    "1. Rangoli banayein. 2. Diye aur fairy lights lagayein. 3. Phoolon ki toran darwaaze par lagayein.",
    "Battery, camera, storage, software updates kitne saal milenge, aur service centre paas mein hai ya nahi, yeh dekhein.",
    "Dosti ka rishta anmol hai yaaron, har mushkil mein saath nibhaana hai yaaron.",
    "Subah ya shaam paani dein, gamle chhaaon mein rakhein, mitti par sookhe patte bichhaayein.",
    "Company ke baare mein padhein, apne resume ke har point par example taiyaar rakhein, aur aam sawaalon ki practice karein.",
    "Maaf kijiye, main is kaam mein madad nahi kar sakta, kyunki yeh gair-kanooni hai aur kisi ko nuksaan pahuncha sakta hai."])]
os.makedirs("/content/out", exist_ok=True)
by = collections.defaultdict(list)
with open("/content/out/lid_calib.jsonl", "w") as f:
    for r in rows:
        preds = lid.f.predict(" ".join(r["text"].split()), 3, 0.0, "strict")  # [(prob, label)]; lid.predict breaks on numpy 2
        probs, labs = [p for p, _ in preds], [l for _, l in preds]
        top = labs[0].replace("__label__", "")
        rec = {**r, "top": top, "p": round(float(probs[0]), 3), "top3": [l.replace("__label__", "") for l in labs],
               "cluster": top.split("_")[0] in H, "cluster_latn": top.split("_")[0] in H and top.endswith("_Latn")}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        by[r["set"]].append(rec)
print(f"{'set':<22} n   cluster  cluster+Latn  labels")
for s, rs in by.items():
    c = sum(x["cluster"] for x in rs); cl = sum(x["cluster_latn"] for x in rs)
    print(f"{s:<22} {len(rs):<3} {c:>3}/{len(rs):<4} {cl:>3}/{len(rs):<6} {dict(collections.Counter(x['top'] for x in rs).most_common(6))}")
print("RUN_OK")
