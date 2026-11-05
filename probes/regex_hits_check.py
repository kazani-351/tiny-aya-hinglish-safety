# Which replies does AuditKIT's English-only RefusalDetector match, and what are they really?
# Inputs (local, gitignored outs/ + committed labels/): outs/auditkit_<cond>.json (AuditKIT probe-16 results,
# scored with refusal_en_regex), outs/rerun_label_key.csv, labels/probe14_rerun_kazani.csv, labels/probe14_rerun_llm_v2.csv.
# Prints counts and ids only. Never prints reply text.
import csv, json, re
from collections import Counter

CONDS = ["base_plain", "base_sys", "base_fewshot", "plainsft"]
key = {(r["cond"], r["id"]): r["row"] for r in csv.DictReader(open("outs/rerun_label_key.csv"))}
kaz = {r["row"]: r for r in csv.DictReader(open("labels/probe14_rerun_kazani.csv"))}
llm = {r["row"]: r for r in csv.DictReader(open("labels/probe14_rerun_llm_v2.csv"))}

print("## A. Every reply the regex matched (all 4 conditions, 32 greedy replies each)")
print("| cond | id | in labelled sample | kazani label / reply language | Claude v2 label / reply language | GlotLID lang_kept |")
print("|---|---|---|---|---|---|")
hits = 0
for c in CONDS:
    for p in json.load(open(f"outs/auditkit_{c}.json"))["predictions"]:
        s = {x["name"]: x["value"] for x in p["metadata"]["scores"]}
        if not s["refusal_en_regex"]:
            continue
        hits += 1
        pid = p["sample_id"].split(":")[1]
        row = key.get((c, pid))
        k = f'{kaz[row]["label"]} / {kaz[row]["reply_language"]}' if row else "-"
        l = f'{llm[row]["label"]} / {llm[row]["reply_language"]}' if row else "-"
        print(f"| {c} | {pid} | {'yes ' + row if row else 'no'} | {k} | {l} | {s['lang_kept']:.0f} |")
print(f"\nTotal regex hits: {hits}")

print("\n## B. Labelled harmful-prompt replies kazani called REFUSE: did the regex catch them?")
print("| kazani reply language | REFUSE rows | regex matched |")
print("|---|---|---|")
scores = {}
for c in CONDS:
    for p in json.load(open(f"outs/auditkit_{c}.json"))["predictions"]:
        scores[(c, p["sample_id"].split(":")[1])] = {x["name"]: x["value"] for x in p["metadata"]["scores"]}["refusal_en_regex"]
tab = Counter(); hit = Counter()
for (c, pid), row in key.items():
    if kaz[row]["label"] == "REFUSE":
        lang = kaz[row]["reply_language"]
        tab[lang] += 1; hit[lang] += int(scores[(c, pid)] == 1.0)
for lang in sorted(tab):
    print(f"| {lang} | {tab[lang]} | {hit[lang]} |")
print(f"| all | {sum(tab.values())} | {sum(hit.values())} |")
