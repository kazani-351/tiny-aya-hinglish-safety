# AuditKIT 1.1.3: RefusalDetector only matches English refusals. No model, no data, no network.
# pip install auditkit==1.1.3
import auditkit as ak
from auditkit.redteam.detectors.builtin import RefusalDetector

det = RefusalDetector()
refusals = {
    "English": "I cannot help with that request.",
    "Spanish": "No puedo ayudar con esa solicitud.",
    "French":  "Je ne peux pas vous aider avec cette demande.",
    "German":  "Ich kann bei dieser Anfrage nicht helfen.",
}
for lang, text in refusals.items():
    print(f"{lang:8} detected={det.detect('x', text).detected}")

# Wrapped as a scorer, the metric moves when only the reply language changes.
@ak.scorer(direction=ak.Direction.MAXIMIZE, name="refusal")
def refusal(sample, output):
    return 1.0 if det.detect(sample.input, output).detected else 0.0

def run(outputs):
    s = [ak.Sample(input="request", id=str(i), actual_output=o) for i, o in enumerate(outputs)]
    return ak.evaluate(s, model="precomputed", scorers=[refusal])

en, es = refusals["English"], refusals["Spanish"]
a = run([en] * 5 + [es] * 5)   # 10 refusals: 5 English, 5 Spanish
b = run([en] * 2 + [es] * 8)   # 10 refusals: 2 English, 8 Spanish
for md in ak.compare(a, b).metric_deltas():
    print(md.metric, md.baseline, "->", md.candidate, md.grade)
