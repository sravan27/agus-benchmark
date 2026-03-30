# Failure Gallery

This gallery highlights the most revealing AGUS failures from the current local-model runs. The goal is not to dump raw traces. The goal is to show short, memorable examples of the kinds of reasoning breakdown AGUS is designed to expose.

## Static-Dynamic Gap

### `attention_distractors_0014` | Llama | `attention_distractors`

- Weakness type: `static_dynamic_gap`, `overconfident_error`, `failed_hypothesis_update`, `poor_attention_recovery`
- Confidence: `1.0`
- Outcome: static task correct, interactive episode incorrect
- Reason: `final_answer_incorrect, missed_contradiction`
- Why this matters: the model solved the frozen version, then failed once the task required live revision after a cue. This is a strong AGUS example because ordinary static accuracy would count it as success.

### `metacog_revision_0010` | Qwen | `metacog_revision`

- Weakness type: `static_dynamic_gap`, `failed_hypothesis_update`, `failed_metacognitive_revision`
- Confidence: `0.0`
- Outcome: static task correct, interactive episode incorrect
- Reason: `final_answer_incorrect, missed_contradiction`
- Why this matters: this is a clean reminder that static-dynamic gaps are not only overconfidence failures. A model can also fail by not updating correctly even when it is initially uncertain.

## Social Belief Confusion

### `social_miniworlds_0003` | Llama | `social_miniworlds`

- Weakness type: `static_dynamic_gap`, `overconfident_error`, `failed_hypothesis_update`, `social_belief_confusion`, `trust_revision_failure`, `deceptive_evidence_failure`
- Confidence: `0.82`
- Outcome: static task correct, interactive episode incorrect
- Reason: `final_answer_incorrect, missed_contradiction`
- Why this matters: the model breaks once AGUS requires separation of world state, agent belief state, and trust under deceptive evidence.

### `social_miniworlds_0015` | Llama | `social_miniworlds`

- Weakness type: `static_dynamic_gap`, `overconfident_error`, `failed_hypothesis_update`, `social_belief_confusion`, `deceptive_evidence_failure`
- Confidence: `0.82`
- Outcome: static task correct, interactive episode incorrect
- Reason: `final_answer_incorrect, missed_contradiction`
- Why this matters: this is a compact judge-facing example of how AGUS can expose social reasoning failure that does not show up in simpler one-shot prompts.

## Overconfident Error

### `attention_distractors_0004` | Llama | `attention_distractors`

- Weakness type: `static_dynamic_gap`, `overconfident_error`, `failed_hypothesis_update`, `poor_attention_recovery`
- Confidence: `0.9`
- Outcome: interactive episode incorrect
- Reason: `final_answer_incorrect, missed_contradiction`
- Why this matters: the failure is not just wrongness. It is wrongness paired with strong confidence after the benchmark has already supplied disambiguating evidence.

### `metacog_revision_0004` | Qwen | `metacog_revision`

- Weakness type: `overconfident_error`, `failed_hypothesis_update`, `failed_metacognitive_revision`
- Confidence: `0.9`
- Outcome: interactive episode incorrect
- Reason: `final_answer_incorrect, missed_contradiction`
- Why this matters: AGUS can still find overconfident failure even in the model that looks better on several trajectory metrics overall.

## Failed Hypothesis Update

### `metacog_revision_0008` | Llama | `metacog_revision`

- Weakness type: `overconfident_error`, `failed_hypothesis_update`, `failed_metacognitive_revision`
- Confidence: `0.95`
- Outcome: interactive episode incorrect
- Reason: `final_answer_incorrect, missed_contradiction`
- Why this matters: the model receives corrective evidence and still does not update its working hypothesis correctly.

### `attention_distractors_0017` | Qwen | `attention_distractors`

- Weakness type: `overconfident_error`, `failed_hypothesis_update`, `poor_attention_recovery`
- Confidence: `0.82`
- Outcome: interactive episode incorrect
- Reason: `final_answer_incorrect, missed_contradiction, failed_attention_recovery`
- Why this matters: even the stronger dynamic model can still get trapped when attention recovery and revision have to happen together.

## Takeaway

These examples are useful because they are interpretable. They show AGUS failing models for reasons a judge can understand:

- the model was right statically but wrong dynamically
- the model stayed overconfident when it should have updated
- the model failed to separate agent beliefs from world state
- the model saw new evidence and still did not revise correctly
