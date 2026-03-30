# Project Name

**AGUS: Adaptive Generalization Under Shift**

## Your Team

**To fill at submission time:** replace this line with the official Kaggle team name and member list.

## Problem Statement

Most reasoning benchmarks are static. They test whether a model can answer a frozen task, but they often miss a more AGI-relevant question: can the model **learn, detect change, revise its hypothesis, and remain coherent after new evidence arrives**?

AGUS is a **Learning-track benchmark** for the Kaggle competition *Measuring Progress Toward AGI - Cognitive Abilities*. Its core thesis is that **static correctness and adaptive reasoning quality can diverge sharply**, so a benchmark for general intelligence should measure both.

## Task & Benchmark Construction

AGUS centers Learning as the primary faculty and uses metacognition, attention, and social cognition as supporting evidence for whether learning is genuinely adaptive rather than shallow.

The benchmark contains five synthetic, human-inspectable task families:

1. `hidden_rule`: infer a latent rule from sparse examples, then adapt after a rule shift.
2. `shift_transfer`: preserve a learned rule across a representation shift.
3. `metacog_revision`: give an answer, confidence, and rule hypothesis, then revise after corrective evidence.
4. `attention_distractors`: identify the true latent structure while resisting salient decoys.
5. `social_miniworlds`: track beliefs, knowledge access, trust, and incentives across short social episodes.

AGUS includes both static tasks and lightweight interactive episodes. In the interactive setting, the model must:

1. observe examples
2. state an initial answer, hypothesis, and confidence
3. receive new evidence, contradiction, or a rule/representation shift
4. revise its answer and hypothesis
5. continue through a short micro-episode when belief tracking or attention recovery matters

This makes the benchmark more diagnostic of **adaptive generalization under shift**, not just one-shot problem solving.

## Dataset

AGUS is generated synthetically with deterministic seeds and a shared schema. Current benchmark components include:

- balanced task exports across five families
- interactive traces for supported task families
- adversarial curation to filter out shallow or shortcut-solvable items
- refinement and search-conditioned refinement loops to improve weak generators over time

The benchmark-development pipeline does not stop at generation. Candidate tasks are challenged by weak baseline probes designed to simulate shallow strategies such as pattern matching, distractor-following, no-revision behavior, and representation anchoring.

## Technical Details

AGUS evaluates more than final accuracy. Key metrics include:

- `accuracy`
- `belief_trajectory_quality`
- `episode_cognitive_flexibility_score`
- `online_adaptation_gain`
- `hypothesis_update_score`
- `attention_recovery_score`
- `belief_state_consistency`
- `trajectory_instability_index`

The benchmark also extracts failure categories such as:

- `static_dynamic_gap`
- `overconfident_error`
- `failed_hypothesis_update`
- `social_belief_confusion`

This lets AGUS produce interpretable evidence about *how* a model fails, not only *whether* it fails.

## Results, Insights, and Conclusions

The first local-model comparisons already support the main AGUS claim.

Balanced 100-task runs on local Ollama models show:

- **Llama 3.1 8B** static accuracy: `0.6179`
- **Qwen 2.5 7B** static accuracy: `0.3000`

But on adaptive reasoning metrics:

- Qwen `belief_trajectory_quality`: `0.7281` versus Llama `0.5434`
- Qwen `episode_cognitive_flexibility_score`: `0.7361` versus Llama `0.6348`
- Qwen `online_adaptation_gain`: `0.525` versus Llama `0.315`
- Qwen `trajectory_instability_index`: `0.2626` versus Llama `0.3348` lower is better

The strongest separating weakness types in the current local runs are:

- `overconfident_error`
- `static_dynamic_gap`
- `social_belief_confusion`

This is the most important result from AGUS so far: **a model can look much stronger on static accuracy while looking weaker on adaptive reasoning quality, revision, and brittleness under changing evidence.**

That is exactly why AGUS fits the Learning track. It measures whether a model can acquire and update task-relevant structure efficiently, not only whether it can answer familiar-looking tasks correctly.

Supporting modules matter here, but they are secondary to the main claim:

- metacognition shows whether the model knows when it may be wrong
- attention shows whether it can resist distractors and recover after clutter
- social cognition shows whether belief updates remain coherent when multiple agents and incentives are involved

Together they help validate that AGUS is measuring **adaptive learning behavior** rather than a narrow synthetic trick.

## Organizational Affiliations

**To fill at submission time:** replace this section with the submitter’s actual affiliation, or state that the work is an independent submission if appropriate.

## References & Citations

1. Google DeepMind. *Measuring progress toward AGI: A cognitive framework.* Mar. 17, 2026. [https://blog.google/innovation-and-ai/models-and-research/google-deepmind/measuring-agi-cognitive-framework/](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/measuring-agi-cognitive-framework/)
2. Burnell, Yamamori, Firat, et al. *Measuring Progress Toward AGI: A Cognitive Framework.* 2026. [https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/measuring-progress-toward-agi/measuring-progress-toward-agi-a-cognitive-framework.pdf](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/measuring-progress-toward-agi/measuring-progress-toward-agi-a-cognitive-framework.pdf)
3. Kaggle competition page. *Measuring Progress Toward AGI - Cognitive Abilities.* [https://www.kaggle.com/competitions/kaggle-measuring-agi](https://www.kaggle.com/competitions/kaggle-measuring-agi)
4. AGUS local result artifacts in this repository:
   - `data/evals/comparisons/llama_vs_qwen_100/comparison_summary.json`
   - `data/evals/comparisons/llama_vs_qwen_100_instability/instability_comparison.json`
   - `data/evals/llama31_balanced_interactive100/distilled_failures.json`
   - `data/evals/qwen25_balanced_interactive100/distilled_failures.json`
