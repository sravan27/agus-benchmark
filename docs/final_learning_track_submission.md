# AGUS: Adaptive Generalization Under Shift

**Sravan Sridhar**

## Problem Statement

I built AGUS out of a frustration with static benchmarking.

Too many model evaluations tell us whether a model can solve a frozen task, but not whether it can **learn from sparse evidence, detect that its old strategy has failed, revise after contradiction, and stay coherent while doing so**. That gap felt especially important in the context of this competition, because the official framing is explicitly cognitive. Google DeepMind’s framework highlights abilities such as learning, metacognition, attention, executive functions, and social cognition, and Kaggle Community Benchmarks exists to let the research community build benchmarks around exactly those harder-to-measure abilities instead of relying only on older static evaluations.  
Sources: [DeepMind cognitive framework](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/measuring-agi-cognitive-framework/), [Kaggle Community Benchmarks](https://blog.google/innovation-and-ai/technology/developers-tools/kaggle-community-benchmarks/), [competition page](https://www.kaggle.com/competitions/kaggle-measuring-agi)

The core thesis behind AGUS is narrow and testable:

**static correctness and adaptive reasoning quality can diverge sharply.**

That is the heart of this submission.

I also want to be precise about scope. **AGUS** is the umbrella benchmark identity for the broader project. The **submitted Kaggle slice** is intentionally narrower: **Learning Core**. I made that choice on purpose. I did not want to submit a sprawling research program and hope judges guessed where the signal was. I wanted to submit the cleanest, strongest Learning-track core of the idea.

## Why I Built It This Way

There is a personal story behind the shape of the benchmark.

I built AGUS independently, under real constraints. I did not have the budget to run a broad frontier-model evaluation campaign with paid API access. The first serious empirical pass had to be built around public artifacts, deterministic generation, and local open-source model evaluation on a **MacBook Pro M2 Max**. That limitation mattered. It pushed the project away from benchmark theater and toward something much more inspectable:

- compact but meaningful interactive episodes
- deterministic benchmark slices
- human-auditable rows
- model-agnostic evaluation logic
- a public repository that exposes not just the final artifact, but the research path that led to it

In hindsight, that constraint helped the project. It forced me to build something that could be defended line by line rather than something that only looked ambitious from far away.

AGUS also did not appear fully formed. It evolved over time. The project began with a narrower question: how do you test whether a model can adapt when the task shifts, rather than merely solve a familiar pattern? That led first to hidden-rule adaptation, then revision under contradiction, then transfer across representation shift. From there the broader repo expanded into distractors, social belief tracking, instability analysis, adversarial curation, and later AGUS v2 counterfactual branching.

The submission is therefore intentionally disciplined: it packages the strongest clean Learning-track core of a broader research evolution.

## Task & Benchmark Construction

The live Kaggle task is:

**`agus_learning_track_v1`**

The submitted Learning Core slice contains exactly three task families:

1. `hidden_rule`: infer a latent rule from sparse examples, then adapt after a rule shift  
2. `shift_transfer`: preserve learned structure across a representation shift  
3. `metacog_revision`: answer, report confidence and a rule hypothesis, then revise after corrective evidence

I chose these three because together they form the clearest expression of the Learning-track claim. They test whether a model can:

- infer structure from minimal evidence
- detect that prior structure no longer explains the task
- preserve latent structure while revising surface form
- articulate a hypothesis and then update it when contradiction appears

The broader AGUS repository contains additional materials, including `attention_distractors`, `social_miniworlds`, instability analysis, curation and refinement machinery, and AGUS v2 counterfactual branching. Those are part of the larger AGUS research program. They are **not** being presented as the submitted Kaggle benchmark itself.

## Dataset

The submitted benchmark is synthetic, deterministic, and human-inspectable.

The Kaggle Learning Core slice contains:

- `10` `hidden_rule` rows
- `10` `shift_transfer` rows
- `10` `metacog_revision` rows

That gives a total of **30 Learning Core rows** in the submitted Kaggle package.

I chose a deterministic, inspectable slice deliberately. The point was not to hide cognition behind scale. The point was to make adaptive behavior visible enough that a reviewer could inspect the benchmark and understand what it was probing.

Submission artifacts:

- **Kaggle task:** [agus_learning_track_v1](https://www.kaggle.com/benchmarks/tasks/sravansridhar27/agus-learning-track-v1)
- **Kaggle benchmark:** [AGUS benchmark project](https://www.kaggle.com/benchmarks/sravansridhar27/agus/versions/1)
- **Kaggle dataset:** [agus-benchmark](https://www.kaggle.com/datasets/sravansridhar27/agus-benchmark)
- **GitHub repository:** [sravan27/agus-benchmark](https://github.com/sravan27/agus-benchmark)

The Kaggle benchmark is the submitted runnable object. The dataset packages the benchmark resources. The GitHub repository is the broader public research record, including the benchmark’s evolution, local evaluation evidence, supporting analyses, and extended AGUS materials.

## Technical Details

The benchmark is implemented as an aggregate Kaggle task over the Learning Core slice. At the evaluation level, AGUS is designed to measure more than final answer accuracy. The most important metrics used in the broader evaluation stack are:

- `accuracy`
- `belief_trajectory_quality`
- `episode_cognitive_flexibility_score`

Supporting analyses in the broader repository also track:

- `trajectory_instability_index`
- `contradiction_blindness_rate`
- failure categories such as `static_dynamic_gap`, `overconfident_error`, and `social_belief_confusion`

That broader analysis matters to me because I did not want AGUS to become just another ranking instrument. I wanted it to help explain **why** models differ, not only **which** model looks best on a frozen score.

## Results, Insights, and Conclusions

The central empirical result behind AGUS is that **static correctness and adaptive reasoning quality separate cleanly**, and that this separation survives fresh deterministic slices.

It is important to state exactly what the headline evidence is. The strongest model-comparison numbers in the repository come from **broader local AGUS runs over a wider five-family suite**, not from the narrower Kaggle Learning Core slice alone. I am stating that plainly because I do not want to blur the difference between the submitted benchmark object and the broader research record.

In those broader local AGUS runs:

- **Llama 3.1 8B** static accuracy: `0.6179`
- **Qwen 2.5 7B** static accuracy: `0.3000`
- **Mistral NeMo 12B** static accuracy: `0.2714`

But adaptive quality does **not** follow the same ranking:

- **Qwen 2.5 7B** `belief_trajectory_quality`: `0.7281`
- **Llama 3.1 8B** `belief_trajectory_quality`: `0.5434`
- **Mistral NeMo 12B** `belief_trajectory_quality`: `0.5828`

That is the main point. In the current broader AGUS evidence, the strongest frozen-task model is **not** the strongest model on adaptive trajectory quality.

Mistral makes the result more credible, not less. It is weak on static accuracy, but not weak everywhere. It performs better on dynamic measures than its static score alone would suggest, and it has the lowest contradiction blindness of the three current local models:

- **Mistral NeMo 12B** `contradiction_blindness_rate`: `0.56`
- **Qwen 2.5 7B** `contradiction_blindness_rate`: `0.64`
- **Llama 3.1 8B** `contradiction_blindness_rate`: `0.66`

The strongest robustness result is that the Llama-versus-Qwen split did **not** disappear on fresh deterministic slices.

On the first deterministic replication slice:

- Llama static accuracy: `0.4857`
- Qwen static accuracy: `0.2857`
- Llama `belief_trajectory_quality`: `0.5606`
- Qwen `belief_trajectory_quality`: `0.7494`

And the same directional split held on **3/3 deterministic replication slices**:

- static accuracy ranking held on `3/3`
- `belief_trajectory_quality` ranking held on `3/3`
- the static-vs-dynamic divergence held on `3/3`
- the main weakness-proxy directions held on `3/3`

That is the single strongest empirical reason I believe this is a strong Learning-track submission. The benchmark is not only conceptually aligned with the competition. It also produces a concrete, replicated result: the model that looks best on static correctness can still be weaker on adaptive reasoning quality.

The project also produces interpretable failure evidence. In the broader AGUS runs, the strongest separating weakness types remain:

- `overconfident_error`
- `static_dynamic_gap`
- `social_belief_confusion`

That matters because I did not want AGUS to be just another benchmark that emits a ranking and stops. I wanted it to point at the *shape* of model failure.

Why do I think this submission merits serious consideration?

Because it does two things at once: it submits a narrow, runnable Learning-track benchmark, and it grounds that benchmark in a broader public research record that shows the idea was developed seriously rather than packaged at the last minute.

- the submitted Kaggle benchmark is live and runnable
- the submitted slice is disciplined and legible
- the benchmark claim is specific
- the broader empirical pattern replicated across fresh deterministic slices
- the public repository shows the benchmark’s evolution, methodology, supporting evidence, and larger AGUS research direction

I wanted the Kaggle submission to stand on its own as a real benchmark, not just as a GitHub project with a story attached. That is why the submitted slice is narrow and runnable. But I also wanted the benchmark to come from a real research process, not appear out of nowhere. That is why the GitHub repository matters: it shows how AGUS evolved, what evidence supports the thesis, and what the broader benchmark program became beyond the submitted slice.

I built AGUS because I think that divergence matters, and because I wanted a benchmark that made it visible instead of hiding it behind a single frozen score.

**Static correctness and adaptive reasoning quality can diverge sharply.**

AGUS is my attempt to measure that divergence directly.

## Organizational Affiliations

Independent submission.

Contact: `sridharsravan@icloud.com`

## References & Citations

1. Google DeepMind. *Measuring progress toward AGI: A cognitive framework.*  
   https://blog.google/innovation-and-ai/models-and-research/google-deepmind/measuring-agi-cognitive-framework/

2. Google. *Introducing Community Benchmarks on Kaggle.*  
   https://blog.google/innovation-and-ai/technology/developers-tools/kaggle-community-benchmarks/

3. Kaggle competition page. *Measuring Progress Toward AGI - Cognitive Abilities.*  
   https://www.kaggle.com/competitions/kaggle-measuring-agi

4. Kaggle task. *agus_learning_track_v1.*  
   https://www.kaggle.com/benchmarks/tasks/sravansridhar27/agus-learning-track-v1

5. Kaggle benchmark. *AGUS benchmark project.*  
   https://www.kaggle.com/benchmarks/sravansridhar27/agus/versions/1

6. Kaggle dataset. *agus-benchmark.*  
   https://www.kaggle.com/datasets/sravansridhar27/agus-benchmark

7. GitHub repository. *sravan27/agus-benchmark.*  
   https://github.com/sravan27/agus-benchmark

8. Repository result artifacts and supporting AGUS analyses, including:
   - `data/evals/llama31_balanced_interactive100/aggregate_summary.json`
   - `data/evals/qwen25_balanced_interactive100/aggregate_summary.json`
   - `data/evals/mistralnemo_balanced_interactive100/aggregate_summary.json`
   - `data/evals/comparisons/llama_qwen_multi_slice_robustness_v1/robustness_summary.json`
   - `data/evals/llama31_counterfactual_v2_expanded/counterfactual_summary.json`
   - `data/evals/qwen25_counterfactual_v2_expanded/counterfactual_summary.json`
   - `data/evals/mistralnemo_counterfactual_v2/counterfactual_summary.json`
