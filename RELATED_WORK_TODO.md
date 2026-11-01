# Related work to-do (for the HF Article)

One line per source: what it gives the article, and the one action left. Read a source in full before citing anything beyond its abstract. Found 2026-11-01 (AISecHub Telegram + web search).

## Must cite (the article is wrong without them)

- [ ] **Kachireddy 2026, "A Measurement Crisis in Multilingual LLM Safety"** (SSRN 10.2139/ssrn.7152058; full PDF in github.com/kaushik0x7d2/indicsafetybench `paper/paper.pdf`). Already showed phrase checks undercount Indic refusals (incl. Hinglish) and that LLM judges fix it. **Use:** credit as the original finding; our part = Tiny Aya + Lexsi's tools + human labels. **Contrast:** his judges count "I cannot understand" as refusal; our rubric labels it OTHER (rule 4). **Action:** quote his exact sentence on code-switching and "both judges count as refusals" with page number.
- [ ] **Zhang, MRL 2026, "Does Safety Live in a Shared Multilingual Subspace?"** (aclanthology.org/2026.mrl-1.19). 10 MultiJail languages, no Hindi/Hinglish; 24-token label-only design cannot see output language (they say so). **Use:** steering/transfer background; cite their own "highest-value follow-up" (control-direction ablation) next to our random-direction control. **Action:** cite the body, not the abstract (the abstract's "0 of nine" is the conjunctive criterion; ablation alone moved six languages).
- [ ] **BabelSteering, arXiv 2608.16577.** English steering vectors raise refusal across 8 languages (+11 pp avg, Gemma 7B) with +13 pp over-refusal. **Use:** the opposing result to Zhang; our over-refusal metric answers the same trade-off. **Action:** check whether they measure output language at all.

## Should cite (from AISecHub)

- [ ] **Adversarial Pragmatics for AI Safety Evaluation, arXiv 2607.01153** (AISecHub #3749). An LLM judge "missed the safety-relevant minority classes"; 4 of 6 agreement statistics could not rule out a constant labeller. **Use:** supports checking the HarmBench judge against hand labels and keeping an OTHER class. **Action:** read sections on the judge assessment; one sentence in Methods.
- [ ] **Refusal to Reframing (China-origin VLMs), arXiv 2608.11816** (AISecHub #4124). Scores refusal and language consistency as separate dimensions; "a model can stop refusing while still reframing". **Use:** precedent for reporting `lang_kept` separately from refusal. **Action:** one sentence in Methods.
- [ ] **HARC, arXiv 2607.00572** (AISecHub #3755). Harmfulness and refusal are separate directions; the model recognises harm while generating it. **Use:** explains why pushing refusal alone can give odd refusals (Zulu attractor, substitution refusals). **Action:** one sentence in Discussion, only if Rule 1 says the drift is refusal-specific.
- [ ] **Breaking Safety at the Token Boundary, arXiv 2607.01239** (AISecHub #3799). Fragmented subword tokens flip first-token refusals on 80-100% of refused HarmBench prompts. **Use:** romanized Hinglish fragments too; context for why Hinglish behaves differently. **Action:** one sentence of context, no claim.
- [ ] **Fahlgren, "How abliteration works"** (calebfahlgren.com, AISecHub #4479). Plain walkthrough of the refusal direction; counts a reply as a refusal if its first word is "I". **Use:** reader-friendly link for the steering section, and one more example of an English-only refusal shortcut. **Action:** link in the steering explainer.

## Context only (do not cite as evidence)

- [ ] **Cohere, "Concerning behavior from a popular open-weight AI model"** (Oct 26, AISecHub #4472). Qwen3.8 attempted an upload to a Beijing bucket in Cohere's sandbox. Not about Aya. **Use:** shows what Cohere's team values (testing models as deployed, provenance). **Action:** none for the article; useful background for the DevRel conversation.
- [ ] **IndicSafe, arXiv 2603.17915; IndicJR (EACL 2026); AILuminate Hindi.** Indic safety benchmarks. **Action:** one line in Related Work listing them; check whether IndicSafe prompts suit a larger rerun.
- [ ] **projectdiscovery, "How abliterated models can get you pwned"** (AISecHub #4405). Not read. **Action:** skip unless the article discusses ablation risk.

## Not useful (checked)

- Devansh-0/Language-Agnostic-Refusal-Vectors: empty repo (README only, 2026-10-19). Re-check before submitting in case results appear.
