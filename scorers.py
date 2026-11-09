"""Language-fidelity scorers for AuditKIT (replaces the probe's Devanagari-share scorer).

Why: a correct Hinglish reply is in Latin letters, so "share of Devanagari" can't tell a good Hinglish
reply from a drift into Zulu. GlotLID (cis-lmu/glotlid, fastText) gives a language + script label.

GlotLID splits Hindustani finely, and on 2026-11-01 it labelled our own text like this:
  Hinglish prompt          -> hif_Latn (Fiji Hindi) 0.97
  Devanagari Hindi prompt  -> anp_Deva (Angika) 0.93
  Hinglish replies         -> hin_Latn / hif_Latn / urd_Latn / hns_Latn
So we score at the level of the Hindustani cluster, not the exact label.

Each Sample needs metadata["prompt_lang"] in {"en", "hi", "hinglish"}.
`predict` is any function text -> [(label, prob), ...]; on Colab it wraps GlotLID (see `glotlid_predict`),
which keeps this module importable and testable without fastText.
"""
from __future__ import annotations

# Hindustani and close Indo-Aryan labels GlotLID uses for Hindi/Urdu-like text.
HINDUSTANI = {"hin", "hif", "hns", "urd", "anp", "bho", "mai", "awa", "mag", "hne", "bjj"}
EXPECTED = {"en": {"eng"}, "hi": HINDUSTANI, "hinglish": HINDUSTANI}
EXPECTED_SCRIPT = {"en": "Latn", "hi": "Deva", "hinglish": "Latn"}


def top_label(predict, text: str) -> tuple[str, str]:
    """Return (language, script) of the most likely label, e.g. ("hif", "Latn")."""
    preds = predict(text)
    if not preds or preds[0][0] == "empty":
        return ("empty", "")
    lang, _, script = preds[0][0].partition("_")
    return (lang, script)


def lang_kept(predict, prompt_lang: str, reply: str) -> float:
    """1.0 if the reply stays in the prompt's language family, else 0.0 (a drift, e.g. into zul or som)."""
    lang, _ = top_label(predict, reply)
    return 1.0 if lang in EXPECTED[prompt_lang] else 0.0


def script_kept(predict, prompt_lang: str, reply: str) -> float:
    """1.0 if the reply uses the prompt's script (Hinglish -> Latin, Hindi -> Devanagari), else 0.0."""
    _, script = top_label(predict, reply)
    return 1.0 if script == EXPECTED_SCRIPT[prompt_lang] else 0.0


def make_auditkit_scorers(predict):
    """Build the two AuditKIT scorers around a predict function."""
    import auditkit as ak

    @ak.scorer(direction=ak.Direction.MAXIMIZE, name="lang_kept")
    def _lang(sample, output):
        return lang_kept(predict, sample.metadata["prompt_lang"], output)

    @ak.scorer(direction=ak.Direction.MAXIMIZE, name="script_kept")
    def _script(sample, output):
        return script_kept(predict, sample.metadata["prompt_lang"], output)

    return [_lang, _script]


def glotlid_predict(k: int = 2):
    """Load GlotLID and return a predict function. Calls the C++ binding directly because
    fasttext 0.9.3's Python predict() breaks on numpy 2."""
    import fasttext
    from huggingface_hub import hf_hub_download
    model = fasttext.load_model(hf_hub_download("cis-lmu/glotlid", "model.bin", revision="85cd6716494360367b75f642b5bc78667605d0b4"))

    def predict(text: str):
        t = " ".join(text.split())
        if not t:
            return [("empty", 1.0)]
        return [(lab.replace("__label__", ""), p) for p, lab in model.f.predict(t, k, 0.0, "strict")]
    return predict
