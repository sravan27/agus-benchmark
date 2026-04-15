# Executive Summary

**AGUS** is a benchmark for **Adaptive Generalization Under Shift**.

The current Kaggle submission packages the **Learning Core** slice: three task families that test whether a model can infer structure from sparse evidence, detect when its current strategy no longer works, and revise after new evidence or a representation shift.

Learning Core includes exactly:

- `hidden_rule`
- `shift_transfer`
- `metacog_revision`

The main result is simple and judge-relevant:

**static correctness and adaptive reasoning quality can diverge sharply.**

In the current local Learning Core runs:

- `llama3.1:8b` leads on static accuracy: `0.6179`
- `qwen2.5:7b` leads on `belief_trajectory_quality`: `0.7281` versus Llama `0.5434`
- `mistral-nemo:12b` reinforces the separation with low static accuracy (`0.2714`) but stronger adaptive quality (`belief_trajectory_quality 0.5828`)

This is not only a one-slice pattern. In the current robustness package, the central Llama-versus-Qwen split held on **3/3 deterministic replication slices**:

- Llama stayed ahead on static accuracy
- Qwen stayed ahead on `belief_trajectory_quality`
- the main weakness-proxy directions also replicated

The repo contains broader AGUS research materials, including `attention_distractors`, `social_miniworlds`, adversarial curation, refinement loops, and AGUS v2 counterfactual branching. Those materials strengthen the research story, but the **submitted Kaggle benchmark package** is the **Learning Core** slice only.

The reviewer takeaway is:

AGUS is not trying to be another broad frozen-task benchmark. It is trying to show that **learning quality under shift** can be measured separately from static correctness, and the current Learning Core evidence supports that claim.
