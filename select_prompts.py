# Pick the 16 study prompts from HarmBench "standard" by a fixed rule (no hand-picking).
# Rule: categories misinformation_disinformation + cybercrime_intrusion; drop code/payload asks
# (hard to translate and to label), drop asks about named or specific real people (defamation),
# keep prompts <= 100 chars; then a seeded random sample of 8 per category.
# Writes HarmBench row indices to selected_ids.json (committed) and the English text to data/ (gitignored).
import json, random, re, sys
SRC = sys.argv[1] if len(sys.argv) > 1 else "outs/harmbench_standard.jsonl"
SEED = 20261101
CODE = re.compile(r"\b(code|script|payload|macro|function|exploit|vulnerabilit|sql|ddos|worm|keylogger)", re.I)
PEOPLE = re.compile(r"\b(specific|fauci|desantis|trump|president|senator|congress)", re.I)
rows = [json.loads(l) for l in open(SRC)]
rng = random.Random(SEED)
picked = []
for cat in ["misinformation_disinformation", "cybercrime_intrusion"]:
    pool = [i for i, r in enumerate(rows) if r["category"] == cat and len(r["prompt"]) <= 100
            and not CODE.search(r["prompt"]) and not PEOPLE.search(r["prompt"])]
    chosen = sorted(rng.sample(pool, 8))
    print(f"{cat}: pool {len(pool)} -> {chosen}")
    picked += chosen
json.dump({"source": "walledai/HarmBench standard", "seed": SEED, "ids": picked}, open("selected_ids.json", "w"), indent=1)
with open("data/prompts_en.jsonl", "w") as f:
    for i in picked:
        f.write(json.dumps({"hb_id": i, "category": rows[i]["category"], "en": rows[i]["prompt"]}) + "\n")
print(f"wrote {len(picked)} prompts")
