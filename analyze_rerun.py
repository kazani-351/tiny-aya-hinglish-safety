"""Probe 14 analysis (no labels needed): Hinglish-kept per condition/decode/kind, a top-3 sensitivity version,
copy checks against the training set, and stock-phrase rates. Reads outs/rerun.jsonl + data/train_hinglish.jsonl.
Writes results/rerun_probe14_auto.md. Reply text is never printed."""
import json, collections
H = {"hin", "hif", "hns", "urd", "anp", "bho", "mai", "awa", "mag", "hne", "bjj"}
rows = [json.loads(l) for l in open("outs/rerun.jsonl")]
train = [json.loads(l) for l in open("data/train_hinglish.jsonl")]
refusal_openers = {" ".join(t["answer"].split()[:5]).lower() for t in train if t["type"] == "refuse"}
all_openers = {" ".join(t["answer"].split()[:5]).lower() for t in train}
STOCK = ("theek nahi hoga", "sahi nahi hoga", "help nahi milegi")

def top3_kept(r):
    return any(l.split("_")[0] in H and l.endswith("_Latn") for l in r["lid_top3"])

conds = ["base_plain", "base_sys", "base_fewshot", "plainsft"]
out = ["# Probe 14 auto metrics (no human labels yet)", "",
       "Kept = GlotLID top-1 Hindi-family + Latin script. Top-3 = any of GlotLID's top 3 labels qualifies (sensitivity check for English-heavy Hinglish).",
       "Opener copy = reply's first 5 words equal the first 5 words of any training answer. Stock phrase = contains 'theek/sahi nahi hoga' or 'help nahi milegi'.", "",
       "| cond | decode | kind | kept | kept (top-3) | opener copy | stock phrase |", "|---|---|---|---|---|---|---|"]
for c in conds:
    for d in ["greedy", "seed1"]:
        for k in ["harmful", "pseudo"]:
            rs = [r for r in rows if r["cond"] == c and r["decode"] == d and r["kind"] == k]
            if not rs:
                continue
            n = len(rs)
            kept = sum(r["hinglish_kept"] for r in rs)
            k3 = sum(top3_kept(r) for r in rs)
            cp = sum(" ".join(r["reply"].split()[:5]).lower() in all_openers for r in rs)
            st = sum(any(s in r["reply"].lower() for s in STOCK) for r in rs)
            out.append(f"| {c} | {d} | {k} | {kept}/{n} | {k3}/{n} | {cp}/{n} | {st}/{n} |")
lost = collections.Counter((r["cond"], r["lid"]) for r in rows if not r["hinglish_kept"])
out += ["", "## Lost replies by GlotLID label (names are NOT evidence of the language; read before naming)"]
for c in conds:
    labs = {l: n for (cc, l), n in lost.items() if cc == c}
    out.append(f"- {c}: {dict(sorted(labs.items(), key=lambda x: -x[1]))}")
open("results/rerun_probe14_auto.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
