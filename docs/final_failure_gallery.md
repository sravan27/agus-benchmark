# Final Failure Gallery

## 1. Static-Dynamic Gap

**Task:** `attention_distractors_0014`  
**Model:** Llama 3.1 8B  
**Why it matters:** static task correct, interactive episode incorrect, confidence `1.0`  
**Interpretation:** the model solved the frozen item but failed once AGUS demanded live attention recovery after new evidence.

## 2. Social Belief Confusion

**Task:** `social_miniworlds_0003`  
**Model:** Llama 3.1 8B  
**Why it matters:** categories include `social_belief_confusion`, `trust_revision_failure`, and `deceptive_evidence_failure`  
**Interpretation:** the model struggled to separate world state, agent belief state, and trust after deceptive evidence.

## 3. Overconfident Error

**Task:** `metacog_revision_0004`  
**Model:** Qwen 2.5 7B  
**Why it matters:** confidence `0.9` with `overconfident_error` and `failed_metacognitive_revision`  
**Interpretation:** even the stronger dynamic model still produces high-confidence misses when contradiction handling breaks.

## 4. Failed Hypothesis Update

**Task:** `metacog_revision_0008`  
**Model:** Llama 3.1 8B  
**Why it matters:** corrective evidence arrives, but the model still fails to revise correctly  
**Interpretation:** this is a clean AGUS example of learning failure under new evidence, not just static wrongness.

## Why These Cases Matter

These examples are concise, legible, and judge-facing. They show that AGUS can expose:

- correct static performance paired with failed dynamic adaptation
- socially grounded belief-state confusion
- harmful overconfidence
- missed hypothesis revision after contradiction
