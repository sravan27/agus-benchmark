# Executive Summary

**AGUS** is a benchmark for **Adaptive Generalization Under Shift**.

The **submitted Kaggle slice** is **Learning Core**, which includes exactly:

- `hidden_rule`
- `shift_transfer`
- `metacog_revision`

The benchmark claim is narrow and evidence-backed:

**static correctness and adaptive reasoning quality can diverge sharply.**

Current Learning Core results:

- `llama3.1:8b` static accuracy: `0.6179`
- `qwen2.5:7b` `belief_trajectory_quality`: `0.7281` versus Llama `0.5434`
- `mistral-nemo:12b` static accuracy: `0.2714`, but `belief_trajectory_quality`: `0.5828`

That result is not confined to the original slice. On the first fresh deterministic replication slice:

- Llama static accuracy: `0.4857`
- Qwen static accuracy: `0.2857`
- Llama `belief_trajectory_quality`: `0.5606`
- Qwen `belief_trajectory_quality`: `0.7494`

And the same Llama-vs-Qwen split held on **3/3 deterministic replication slices**.

The broader repo contains additional AGUS materials, including `attention_distractors`, `social_miniworlds`, and AGUS v2 counterfactual branching. Those are supporting research materials. The current Kaggle package is the **Learning Core** slice only.

The takeaway is simple:

AGUS is trying to measure whether a model can **learn under shift**, not just solve frozen tasks. The current Learning Core evidence supports that claim.
