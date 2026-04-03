# AGUS Hostile Review Defense v1

This is an internal red-team memo. The point is not to make AGUS sound impressive. The point is to identify the easiest ways a skeptical judge could dismiss it and state the strongest defense the current repo can honestly support.

## 1. "This is just another benchmark."

**Possible judge criticism**

AGUS is only a new synthetic benchmark wrapper around familiar ideas like rule induction, reasoning, and transfer.

**Why that criticism is plausible**

Many benchmark submissions rename known task types, add custom metrics, and claim novelty without changing what is really being measured.

**Best available defense from our current benchmark and evidence**

AGUS is not only a static dataset. It combines:

- static tasks
- interactive revision episodes
- instability analysis
- adversarial curation against shallow probes
- search-conditioned refinement
- AGUS v2 counterfactual branching episodes

The most defensible novelty is not any one task family. It is the benchmark logic: AGUS measures whether models update coherently under shift, contradiction, distractors, and nearby alternate futures.

**Status**

Partially answered.

The defense is real, but a hostile reviewer could still say the components are incremental rather than a wholly new category.

## 2. "The result is too narrow."

**Possible judge criticism**

The whole argument may hinge on a narrow artifact of a few synthetic tasks and a few local models.

**Why that criticism is plausible**

Current empirical evidence is from three open local models, and AGUS is still synthetic.

**Best available defense from our current benchmark and evidence**

The three-model set matters. The core AGUS pattern is no longer just Llama versus Qwen:

- Llama leads frozen-task accuracy
- Qwen leads adaptive trajectory quality
- Mistral has weak static accuracy but lower contradiction blindness and stronger counterfactual update behavior

That makes the central claim less likely to be a single pairwise fluke. We are not claiming broad generality. We are claiming that AGUS can reveal separable axes that static score alone hides.

**Status**

Partially answered.

This is better than a two-model story, but still a small evidence base.

## 3. "The benchmark is too broad."

**Possible judge criticism**

AGUS tries to do learning, metacognition, attention, social cognition, instability, and counterfactuals all at once, which may make the thesis feel diffuse.

**Why that criticism is plausible**

Submissions that cover too many faculties can look like collections of interesting ideas rather than a tight benchmark contribution.

**Best available defense from our current benchmark and evidence**

The repo now positions AGUS explicitly as a **Learning-track** benchmark. Metacognition, attention, and social cognition are supporting modules, not the main claim. The unifying question is the same across families: can the model learn, notice change, revise, and stay coherent under shift?

AGUS v2 stays aligned with that same thesis by testing coherence across nearby alternate futures rather than introducing a separate benchmark identity.

**Status**

Partially answered.

The framing is now tighter, but a reviewer may still prefer a narrower first submission.

## 4. "Local-model comparisons are weak evidence."

**Possible judge criticism**

Results on Ollama-hosted open models are not strong enough to support meaningful AGI-benchmark claims.

**Why that criticism is plausible**

A judge could reasonably expect frontier evidence, broader baselines, or at least more models before trusting the pattern.

**Best available defense from our current benchmark and evidence**

We should not defend this by pretending local models are enough to prove frontier relevance. The actual defense is narrower:

- local models are sufficient to test whether the benchmark produces interpretable separation
- AGUS already shows non-trivial rank reversals across static, dynamic, instability, and counterfactual metrics
- the benchmark is being evaluated as a **Learning-track benchmark design**, not as a definitive model leaderboard

In other words, the evidence is strong for benchmark usefulness, not for broad model-ranking conclusions.

**Status**

Partially answered.

This remains a real limitation and must stay visible in the writeup.

## 5. "Dynamic metrics are too interpretive."

**Possible judge criticism**

Metrics like `belief_trajectory_quality`, `trajectory_instability_index`, or `counterfactual_update_fidelity` may feel hand-designed and hard to trust.

**Why that criticism is plausible**

Custom benchmark metrics can look subjective, especially when they aggregate several behavioral signals.

**Best available defense from our current benchmark and evidence**

The current defense is that AGUS does not rely on one opaque score. It exposes multiple metric families and keeps traces inspectable:

- exact static accuracy is still reported
- per-turn interactive traces are saved
- failure cases are distilled into concrete categories
- instability metrics explicitly mark directionality
- counterfactual metrics are tied to controlled branch invariants

Also, the most persuasive AGUS evidence is not the existence of one composite metric. It is the repeated separation across several independent measurements.

**Status**

Partially answered.

The metrics are inspectable, but some judges may still want stronger ablation or human validation.

## 6. "AGUS v2 is interesting but under-evaluated."

**Possible judge criticism**

Counterfactual branching is the most novel part, but it may not yet have enough empirical depth to carry real weight.

**Why that criticism is plausible**

The current branch bundle set is compact, and `cross_branch_consistency` saturates at `1.0` for all three models.

**Best available defense from our current benchmark and evidence**

We should present AGUS v2 as an extension that strengthens the benchmark thesis, not as a fully mature standalone benchmark. Even with the compact bundle set, AGUS v2 already separates models on:

- `counterfactual_update_fidelity`
- `invariant_preservation_score`
- `branch_belief_coherence`
- `counterfactual_confidence_calibration`

That is enough to argue benchmark promise and conceptual novelty, not enough to claim v2 is fully validated.

**Status**

Partially answered.

This is one of the clearest remaining weaknesses.

## 7. "The writeup overstates what the evidence supports."

**Possible judge criticism**

The submission may imply stronger conclusions than the current evidence can bear.

**Why that criticism is plausible**

This is a common failure mode in benchmark writeups, especially when the benchmark design is stronger than the empirical section.

**Best available defense from our current benchmark and evidence**

The current submission materials now keep the main caveats visible:

- three local models, not frontier scale
- synthetic tasks
- all three models still substantially contradiction-blind
- AGUS v2 uses a compact counterfactual set

The safest headline is also disciplined: AGUS shows that frozen-task accuracy, adaptive reasoning quality, and counterfactual coherence are separable dimensions. That is a benchmark claim, not a broad intelligence claim.

**Status**

Mostly answered, if we stay disciplined.

This becomes a real weakness again the moment we drift into broader model claims.

## 8. "Model rankings are unstable or sample-dependent."

**Possible judge criticism**

Different seeds, prompts, task subsets, or run sizes might change the ordering enough that the result becomes unreliable.

**Why that criticism is plausible**

Synthetic benchmarks and local inference stacks can be sensitive to prompt details and sample composition.

**Best available defense from our current benchmark and evidence**

The repo does a meaningful amount to reduce accidental instability:

- deterministic generation
- balanced family sampling
- resumable structured evaluation artifacts
- adversarial curation
- refinement and search-conditioned refinement to reduce weak templates

The benchmark also now has a direct lightweight replication result on a **fresh deterministic balanced slice** for the main Llama-versus-Qwen comparison:

- original slice: Llama static accuracy `0.6179` versus Qwen `0.3000`; Qwen `belief_trajectory_quality` `0.7281` versus Llama `0.5434`
- replication slice: Llama static accuracy `0.4857` versus Qwen `0.2857`; Qwen `belief_trajectory_quality` `0.7494` versus Llama `0.5606`
- the main weakness-proxy directions also replicated

Also, the main result is not "Qwen beats Llama" or "Mistral beats Qwen." The main result is that **different metrics induce different rankings**. That claim is more robust to local rank swapping than a single leaderboard claim would be.

**Status**

Mostly answered for the one-slice-fluke criticism.

This is still not a full robustness story. We only have one fresh deterministic replication slice, and the evidence remains local-model-scoped.

## 9. "This is mostly engineering, not science."

**Possible judge criticism**

AGUS may look like an impressive pipeline, but not a scientific contribution.

**Why that criticism is plausible**

The repo includes a lot of machinery: curation, refinement, search, local eval, failure distillation, instability analysis, counterfactual runners.

**Best available defense from our current benchmark and evidence**

The pipeline is not the scientific claim. The scientific claim is that **adaptive reasoning quality is not reducible to frozen-task correctness**, and AGUS makes that visible in a controlled way. The engineering only matters because it supports that claim:

- curation reduces shortcut-friendly tasks
- refinement uses weak-solver failure to improve generation
- interactive evaluation measures revision under evidence
- counterfactual branching measures coherence across nearby alternate futures

If presented correctly, the engineering is evidence that the benchmark was stress-tested, not the core contribution itself.

**Status**

Partially answered.

A judge who dislikes benchmark-engineering submissions may still hold this against us.

## 10. "The benchmark could still be gameable."

**Possible judge criticism**

Even if AGUS is more dynamic than a static set, models may still exploit recurring templates, prompt formatting, or shallow branch heuristics.

**Why that criticism is plausible**

Synthetic benchmarks are always exposed to shortcut risk.

**Best available defense from our current benchmark and evidence**

AGUS is stronger here than most synthetic benchmark submissions because it does not stop at generation:

- weak baseline probes explicitly test shallow strategies
- adversarial curation rejects shortcut-solvable tasks
- refinement analysis pushes generator changes back upstream
- probe-conditioned search selects stronger generator settings

That does not prove AGUS is ungameable. It does show that shortcut resistance is part of the benchmark design itself, not an afterthought.

**Status**

Partially answered.

This remains a real weakness in principle, but it is also one of AGUS's better-defended design areas.

## What We Must NOT Claim

- We must not claim AGUS has already demonstrated frontier-scale validity.
- We must not claim one model is the strongest overall.
- We must not claim the current counterfactual results prove robust coherence in an absolute sense.
- We must not claim the dynamic metrics are fully validated psychological measurements.
- We must not imply that three local models settle the question of generality.

## What We Can Safely Claim

- AGUS measures something importantly different from frozen-task accuracy alone.
- Across three local models, AGUS reveals separable axes: static correctness, adaptive trajectory quality, instability, and counterfactual coherence.
- The three-model pattern is more credible than the earlier two-model contrast.
- AGUS v2 is conceptually strong and already empirically suggestive, even if still under-evaluated.
- The benchmark pipeline takes shortcut risk seriously through curation, refinement, and probe-conditioned search.

## What Would Most Strengthen This Submission If We Had One More Day

- Run one or two stronger additional open local models and check whether the three-axis separation persists.
- Expand AGUS v2 with a larger counterfactual bundle set so the branching result looks less preliminary.
- Add one small metric-validation section showing how the most important dynamic metrics correspond to clear trace-level behaviors.
- Add one compact ablation showing that adversarial curation materially improves retained benchmark signal.
- Produce one clean figure that makes the three-way separation instantly legible: static accuracy, adaptive trajectory quality, and counterfactual coherence.
