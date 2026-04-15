# Project Name

**AGUS: Adaptive Generalization Under Shift**

## Your Team

**Sridhar Sravan**  
Independent submission

## Problem Statement

I came to this competition with a simple frustration: many model benchmarks are static. They tell us whether a model can answer a frozen task, but they often fail to tell us whether the model can **learn, detect that its old strategy is no longer valid, revise after new evidence, and stay coherent while doing so**.

That frustration lined up directly with the official framing of this competition. Google DeepMind’s cognitive framework explicitly calls out abilities like **learning, metacognition, attention, executive function, and social cognition**, and the Kaggle Community Benchmarks platform exists to let the research community build evaluations around those harder-to-measure abilities instead of only relying on older frozen benchmarks.  
Sources: [DeepMind cognitive framework](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/measuring-agi-cognitive-framework/), [Kaggle Community Benchmarks](https://blog.google/innovation-and-ai/technology/developers-tools/kaggle-community-benchmarks/), [competition page](https://www.kaggle.com/competitions/kaggle-measuring-agi)

So the core idea behind AGUS was to benchmark models **dynamically, not statically**.

The main thesis is narrow and testable:

**static correctness and adaptive reasoning quality can diverge sharply.**

## Task & Benchmark Construction

**AGUS** is the umbrella benchmark identity for the repository.  
The **submitted Kaggle benchmark slice** is **Learning Core**.

Learning Core contains exactly three task families:

1. `hidden_rule`: infer a latent rule from sparse examples, then adapt after a rule shift.
2. `shift_transfer`: preserve a learned rule across a representation shift.
3. `metacog_revision`: give an answer, confidence, and rule hypothesis, then revise after corrective evidence.

I chose these three because they are the cleanest expression of the Learning-track claim. Together they test whether a model can:

- infer task structure from minimal evidence
- notice that previous structure no longer explains the data
- maintain latent structure while revising surface form
- express a hypothesis and then update it when contradiction appears

The broader AGUS repository also contains `attention_distractors`, `social_miniworlds`, adversarial curation, refinement loops, instability analysis, and AGUS v2 counterfactual branching. Those broader materials are part of the research program, but they are not the same thing as the submitted Kaggle slice.

## Dataset

The benchmark is synthetic, deterministic, and human-inspectable.

For the submitted Kaggle package, Learning Core contains:

- `10` `hidden_rule` tasks
- `10` `shift_transfer` tasks
- `10` `metacog_revision` tasks

That gives a total submitted Kaggle slice of **30 Learning Core benchmark rows**, packaged in the Kaggle benchmark implementation here:

- benchmark project: [agus-learning-core-v1](https://www.kaggle.com/benchmarks/sravansridhar27/agus-learning-core-v1)
- underlying task: [agus-learning-track-v1 task](https://www.kaggle.com/benchmarks/tasks/sravansridhar27/agus-learning-track-v1/1)

The wider repository also includes larger generated pools, interactive traces, curation artifacts, refinement outputs, and local-model evaluation artifacts. Those are supporting materials that make the benchmark easier to inspect and stress-test.

## Technical Details

AGUS measures more than final accuracy. The most important Learning Core metrics in the current evidence are:

- `accuracy`
- `belief_trajectory_quality`
- `episode_cognitive_flexibility_score`

Supporting analyses in the repo also track:

- `trajectory_instability_index`
- `contradiction_blindness_rate`
- failure types such as `static_dynamic_gap`, `overconfident_error`, and `social_belief_confusion`
- AGUS v2 counterfactual metrics such as `counterfactual_update_fidelity` and `branch_belief_coherence`

There is also a personal reason the benchmark took this shape. I built and evaluated AGUS on a **MacBook Pro M2 Max** and did not have budget for paid API tokens, so the first full empirical pass had to be done with **open-source local models**. That constraint mattered. It pushed the benchmark toward:

- lightweight but meaningful interactive episodes
- deterministic generation and reproducible slices
- transparent artifacts that can be verified in GitHub and Kaggle
- model-agnostic evaluation rather than vendor-specific prompting

In hindsight, that limitation helped. It forced the benchmark to be inspectable and runnable rather than just ambitious on paper.

All core benchmark code, Kaggle packaging, local evaluation artifacts, replication outputs, and supporting research materials are public in the repository:

- GitHub: [sravan27/agus-benchmark](https://github.com/sravan27/agus-benchmark)

## Results, Insights, and Conclusions

The main AGUS result is already visible in the current Learning Core runs.

On the original balanced Learning Core slice:

- **Llama 3.1 8B** static accuracy: `0.6179`
- **Qwen 2.5 7B** static accuracy: `0.3000`
- **Mistral NeMo 12B** static accuracy: `0.2714`

But adaptive quality does **not** follow the same ranking:

- **Qwen 2.5 7B** `belief_trajectory_quality`: `0.7281`
- **Llama 3.1 8B** `belief_trajectory_quality`: `0.5434`
- **Mistral NeMo 12B** `belief_trajectory_quality`: `0.5828`

That means the strongest frozen-task model in the current local set is **not** the strongest model on adaptive trajectory quality.

Mistral makes the result more credible, not less. It acts as a third anchor. Mistral is weak on static accuracy, but it is not simply weak everywhere. It is stronger than its static score would suggest on dynamic behavior, and it has the lowest contradiction blindness of the three current local models:

- **Mistral NeMo 12B** `contradiction_blindness_rate`: `0.56`
- **Qwen 2.5 7B** `contradiction_blindness_rate`: `0.64`
- **Llama 3.1 8B** `contradiction_blindness_rate`: `0.66`

The most important robustness result is that the main Llama-versus-Qwen split did **not** disappear on a fresh slice.

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

That is the single strongest empirical reason I think this is a competitive submission. It is no longer just “Llama and Qwen look different.” It is a benchmark result that:

- has a clear cognitive thesis
- is implemented as a real Kaggle benchmark
- has public code and artifacts
- survives fresh deterministic slices

AGUS v2 is a supporting extension, not the main submission claim, but it strengthens the overall story. It asks whether a model stays coherent across nearby alternate futures. On current counterfactual branch runs:

- **Mistral** is strongest on `counterfactual_update_fidelity`: `0.8889`
- **Mistral** is strongest on `invariant_preservation_score`: `0.9375`
- **Qwen** is strongest on `branch_belief_coherence`: `0.8542`
- **Qwen** is strongest on `counterfactual_confidence_calibration`: `0.9717`

I do not think AGUS v2 should be oversold as a fully mature benchmark on its own yet. But I do think it matters because it shows the same core idea extending one step further: not just “can the model revise,” but “can the model stay coherent across nearby alternate futures?”

The project also produces interpretable failure evidence. The strongest separating weakness types in the current runs remain:

- `overconfident_error`
- `static_dynamic_gap`
- `social_belief_confusion`

That matters because I do not want this submission to be “another leaderboard with custom metrics.” I want it to be a benchmark that helps explain **why** models differ, not only **which** model wins.

Why do I think this is a strong Learning-track submission?

Because it is aligned with the official competition thesis, but it also stays concrete:

- the benchmark is live on Kaggle
- the submitted slice is narrow and legible
- the main claim is specific
- the main result replicated
- the supporting repository makes the work auditable

The broader AGUS repo shows how the idea grew: from hidden-rule adaptation, to revision, to representation shift, to distractors, to social belief tracking, to counterfactual branching. But the submitted Kaggle slice remains disciplined. It packages the strongest and cleanest Learning-track core.

If judges want to go deeper, the GitHub repository contains the full benchmark logic, result artifacts, Kaggle packaging, and supporting research materials. If they only read the writeup and click the benchmark, the main story should still stand on its own.

## Organizational Affiliations

Independent submission.  
Questions or clarifications: `sridharsravan@icloud.com`

## References & Citations

1. Google DeepMind. *Measuring progress toward AGI: A cognitive framework.* Mar. 17, 2026. [https://blog.google/innovation-and-ai/models-and-research/google-deepmind/measuring-agi-cognitive-framework/](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/measuring-agi-cognitive-framework/)
2. Google. *Introducing Community Benchmarks on Kaggle.* Jan. 14, 2026. [https://blog.google/innovation-and-ai/technology/developers-tools/kaggle-community-benchmarks/](https://blog.google/innovation-and-ai/technology/developers-tools/kaggle-community-benchmarks/)
3. Kaggle competition page. *Measuring Progress Toward AGI - Cognitive Abilities.* [https://www.kaggle.com/competitions/kaggle-measuring-agi](https://www.kaggle.com/competitions/kaggle-measuring-agi)
4. Kaggle benchmark project. *AGUS Learning Core v1.* [https://www.kaggle.com/benchmarks/sravansridhar27/agus-learning-core-v1](https://www.kaggle.com/benchmarks/sravansridhar27/agus-learning-core-v1)
5. Kaggle task page. *AGUS Learning Track v1.* [https://www.kaggle.com/benchmarks/tasks/sravansridhar27/agus-learning-track-v1/1](https://www.kaggle.com/benchmarks/tasks/sravansridhar27/agus-learning-track-v1/1)
6. GitHub repository. *sravan27/agus-benchmark.* [https://github.com/sravan27/agus-benchmark](https://github.com/sravan27/agus-benchmark)
7. AGUS result artifacts in this repository, including:
   - `data/evals/llama31_balanced_interactive100/aggregate_summary.json`
   - `data/evals/qwen25_balanced_interactive100/aggregate_summary.json`
   - `data/evals/mistralnemo_balanced_interactive100/aggregate_summary.json`
   - `data/evals/comparisons/llama_qwen_multi_slice_robustness_v1/robustness_summary.json`
   - `data/evals/llama31_counterfactual_v2_expanded/counterfactual_summary.json`
   - `data/evals/qwen25_counterfactual_v2_expanded/counterfactual_summary.json`
   - `data/evals/mistralnemo_counterfactual_v2/counterfactual_summary.json`
