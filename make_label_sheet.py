# Build kazani's blind labelling sheet for Rule 6: probe-4 harmful replies from `base` and `en_L15` (64 total).
# to_label.csv = id + reply text only (shuffled, seeded). label_key.csv = id -> condition/lang/hb_id/judge/lid. Both in outs/ (gitignored).
# Do NOT open label_key.csv until every row in to_label.csv has a label.
import csv, json, random, sys
SRC = sys.argv[1] if len(sys.argv) > 1 else "outs/downstream_replies.jsonl"
labels = {(l["cond"], l["lang"], l["hb_id"]): l for l in map(json.loads, open("outs/downstream_labels.jsonl"))}
rows = [r for r in map(json.loads, open(SRC)) if r["kind"] == "harm" and r["cond"] in ("base", "en_L15")]
assert len(rows) == 64, len(rows)
random.Random(20261101).shuffle(rows)
with open("outs/to_label.csv", "w", newline="") as f, open("outs/label_key.csv", "w", newline="") as k:
    w = csv.writer(f); kw = csv.writer(k)
    w.writerow(["id", "reply", "label (REFUSE/COMPLY/OTHER)", "reply_language", "unsure (yes/blank)", "translated (yes/blank)"])
    kw.writerow(["id", "cond", "lang", "hb_id", "judge", "lid"])
    for i, r in enumerate(rows, 1):
        rid = f"R{i:02d}"
        w.writerow([rid, r["reply"], "", "", "", ""])
        l = labels[(r["cond"], r["lang"], r["hb_id"])]
        kw.writerow([rid, r["cond"], r["lang"], r["hb_id"], l["judge"], r["lid"]])
print("wrote outs/to_label.csv (64 rows) and outs/label_key.csv")
