# Nov 3 probe 16 (CPU, no generation): the before/after through AuditKIT, so the Track 2 chain is real:
#   SafeTune (PlainSFT adapter) -> AuditKIT (ak.evaluate on precomputed replies + ak.compare).
# Scores probe 14's saved greedy replies (32 prompts x 4 conditions) with:
#   lang_kept, script_kept  (our scorers.py, GlotLID inside AuditKIT's scorer API)
#   refusal_en_regex        (AuditKIT's own RefusalDetector, kept as tool-feedback evidence: it is English-only)
# Then ak.compare(base_plain, X) for each repair. No reply text is printed.
# Run: OUT=/content/out UPLOAD=outs/auditkit_bundle.tar EXEC_TIMEOUT=2400 ./run_cpu.sh probes/auditkit_beforeafter.py lexsi-ak
# Bundle = rerun.jsonl + scorers.py. auditkit evaluate(model="precomputed") makes no network calls (source checked 2026-11-03).
import subprocess, sys, os, json, tarfile, time
t0 = time.perf_counter()
if os.path.exists("/content/hf.env"):
    os.remove("/content/hf.env")  # runner always uploads the token; GlotLID is ungated, AuditKIT precomputed needs none
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "auditkit==1.1.3", "fasttext==0.9.3"], capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-500:]
with tarfile.open("/content/auditkit_bundle.tar") as tf:
    tf.extractall("/content/bundle")
sys.path.insert(0, "/content/bundle")
os.makedirs("/content/out", exist_ok=True)  # run 1 (2026-11-03) died here: every probe must create it
import auditkit as ak
from auditkit.redteam.detectors.builtin import RefusalDetector
from scorers import make_auditkit_scorers, glotlid_predict
print(f"auditkit {getattr(ak, '__version__', '?')} | setup {time.perf_counter()-t0:.0f}s", flush=True)

predict = glotlid_predict(k=2)
lang, script = make_auditkit_scorers(predict)
det = RefusalDetector()
@ak.scorer(direction=ak.Direction.MAXIMIZE, name="refusal_en_regex")
def refusal_en(sample, output):
    return 1.0 if det.detect(sample.input, output).detected else 0.0

rows = [json.loads(l) for l in open("/content/bundle/rerun.jsonl") if json.loads(l)["decode"] == "greedy"]
conds = ["base_plain", "base_sys", "base_fewshot", "plainsft"]
results, out = {}, {}
for c in conds:
    samples = [ak.Sample(input=r["prompt"], id=f"{r['kind']}:{r['id']}", actual_output=r["reply"],
                         metadata={"prompt_lang": "hinglish", "kind": r["kind"]}) for r in rows if r["cond"] == c]
    assert len(samples) == 32, (c, len(samples))
    res = ak.evaluate(samples, model="precomputed", scorers=[lang, script, refusal_en])
    results[c] = res
    out[c] = {"headline": str(res.headline), "errors": len(res.errors), "failed": res.failed_count}
    print(f"[{c}] headline={res.headline} errors={len(res.errors)} failed={res.failed_count}", flush=True)
    try:
        res.save(f"/content/out/auditkit_{c}.json", fmt="json", write_provenance=True)
    except Exception as e:
        print(f"  save failed: {e!r}"[:200])
for c in conds[1:]:
    try:
        cmp = ak.compare(results["base_plain"], results[c])
        txt = cmp.summary() or ""  # run 1 printed an empty string; also record the raw deltas
        txt += "\n" + "\n".join(repr(md) for md in cmp.metric_deltas()) + "\n" + repr(cmp.sample_summary())
        print(f"\n=== ak.compare(base_plain, {c}) ===\n{txt}", flush=True)
        out[f"compare_{c}"] = txt
    except Exception as e:
        print(f"compare {c} failed: {e!r}"[:300])
        out[f"compare_{c}"] = f"failed: {e!r}"[:300]
with open("/content/out/auditkit_beforeafter.json", "w") as f:
    json.dump(out, f, indent=1)
print(f"total {time.perf_counter()-t0:.0f}s")
print("RUN_OK")
