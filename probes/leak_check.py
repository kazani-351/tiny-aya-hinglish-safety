# Train/test leak check: does any eval prompt appear in the 42 training rows, exactly or nearly?
# Usage: python3 probes/leak_check.py <hf_dataset_dir> [<data_dir with prompts.jsonl>]
#   hf_dataset_dir: the published HF dataset (train_hinglish.csv, pseudo_harmful_hinglish.csv)
#   data_dir: optional; the translated harmful prompts (data/prompts.jsonl) are withheld, so
#             without it only the over-refusal half runs.
# Checks every train x eval pair: exact match after lowercasing and stripping punctuation,
# character 5-gram Jaccard, word 3-gram Jaccard (exact Jaccard, which MinHash only estimates),
# and difflib's sequence ratio. Stdlib only, CPU, under a second.
import csv, difflib, json, os, re, sys, unicodedata

hf = sys.argv[1]
data = sys.argv[2] if len(sys.argv) > 2 else None

def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s)).strip()

def char_grams(s, n=5):
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}

def word_grams(s, n=3):
    w = s.split()
    return {tuple(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}

def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0

train = [(r["id"], r["prompt"]) for r in csv.DictReader(open(f"{hf}/train_hinglish.csv"))]
evals = {"overrefusal": [(r["id"], r["prompt"]) for r in csv.DictReader(open(f"{hf}/pseudo_harmful_hinglish.csv"))]}
if data and os.path.exists(f"{data}/prompts.jsonl"):
    evals["harmful"] = [(f"hb{r['hb_id']}", r["hinglish"]) for r in map(json.loads, open(f"{data}/prompts.jsonl"))]
else:
    print("harmful eval set not found (withheld); checking the over-refusal set only")

print(f"train={len(train)} " + " ".join(f"{k}={len(v)}" for k, v in evals.items()))
rows = []
for name, ev in evals.items():
    exact = [(e, t) for e, ep in ev for t, tp in train if norm(ep) == norm(tp)]
    print(f"{name}: exact matches after normalising = {len(exact)}")
    for e, ep in ev:
        for t, tp in train:
            a, b = norm(ep), norm(tp)
            rows.append((jaccard(char_grams(a), char_grams(b)), jaccard(word_grams(a), word_grams(b)),
                         difflib.SequenceMatcher(None, a, b).ratio(), name, e, t))
for k, label in ((0, "char 5-gram Jaccard"), (1, "word 3-gram Jaccard"), (2, "difflib ratio")):
    m = max(rows, key=lambda r: r[k])
    print(f"max {label} = {m[k]:.3f} ({m[3]} {m[4]} vs {m[5]})")
print(f"pairs with char5 >= 0.5: {sum(r[0] >= 0.5 for r in rows)}, word3 >= 0.3: {sum(r[1] >= 0.3 for r in rows)}, "
      f"difflib >= 0.8: {sum(r[2] >= 0.8 for r in rows)}, of {len(rows)} pairs")
print("top 5 pairs by char 5-gram Jaccard (char5, word3, difflib):")
for r in sorted(rows, key=lambda r: -r[0])[:5]:
    print(f"  {r[0]:.2f} {r[1]:.2f} {r[2]:.2f}  {r[3]} {r[4]} vs {r[5]}")
