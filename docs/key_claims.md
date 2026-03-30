# Key Claims

## Three Strongest Claims We Can Safely Make

1. AGUS distinguishes static accuracy from dynamic reasoning quality. In the current local runs, `llama3.1:8b` leads on static accuracy (`0.6179` versus `0.3000`), while `qwen2.5:7b` leads on `belief_trajectory_quality` (`0.7281` versus `0.5434`) and `episode_cognitive_flexibility_score` (`0.7361` versus `0.6348`).
2. AGUS surfaces meaningful instability differences that static score alone would hide. Llama shows higher overall `trajectory_instability_index` (`0.3348` versus `0.2626`), higher `unnecessary_revision_rate` (`0.30` versus `0.11`), and higher `brittle_reversal_rate` (`0.32` versus `0.11`).
3. The most judge-legible weakness separations in the current local runs are `overconfident_error`, `static_dynamic_gap`, and `social_belief_confusion`, all of which appear substantially more often for Llama than for Qwen.

## Three Claims We Should Not Overstate Yet

1. We should not claim that `qwen2.5:7b` is the stronger model overall. It is stronger on several AGUS dynamic metrics, but much weaker on static accuracy.
2. We should not claim that AGUS has already validated dynamic cognition at frontier scale. The current evidence is from two local open models, not a broad frontier-model comparison.
3. We should not claim that either model handles contradiction well. Both still have high `contradiction_blindness_rate` values, so the current result is relative, not absolute.

## Three Important Limitations

1. The evidence comes from a small local-model set with one primary inference stack per model, so prompt and adapter choices may matter.
2. AGUS is synthetic and interactive, which is a strength for controllability but still leaves open questions about transfer to more naturalistic tasks.
3. The current results packet is strongest on comparative patterns and failure signatures, not on broad claims about real-world deployment competence.
