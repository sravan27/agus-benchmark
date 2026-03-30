# AGUS Benchmark Design

## Motivation

Many benchmark suites are static. They largely measure whether a model has seen similar tasks before, or whether it can perform one-shot reasoning inside a frozen problem frame. AGUS takes a different view: a strong AGI-oriented benchmark should stress **dynamic cognition**.

AGUS therefore emphasizes:

- sample-efficient rule learning
- adaptation after change
- transfer under surface remapping
- explicit uncertainty management
- revision after corrective evidence
- selective attention under distractors

## Design Principles

### 1. Dynamic over static

Tasks should create a meaningful temporal structure: learn, detect change, revise, transfer, or ignore noise.

### 2. Human inspectability

Every task is synthetic but human-readable. A person should be able to inspect examples and understand why a task is difficult.

### 3. Extensibility

The generators use a shared schema and deterministic seeds so families can scale in breadth and difficulty without losing coherence.

### 4. Legitimacy over gimmicks

We avoid prompt-format exploits and pure leaderboard tricks. The benchmark aims to support a research argument about adaptive cognition.

## Task Families

## A. Hidden Rule Induction

The model observes a small number of input-output examples and must infer the latent transformation. After initial induction, the rule changes. The model then receives a small amount of new evidence and must recover quickly.

Primary signals:

- sample efficiency
- change detection
- adaptation speed

Failure mode exposed:

- brittle reliance on a cached hypothesis

## B. Shift and Transfer

The model learns a rule in one symbolic language, then must apply the same latent structure after the surface representation changes.

Primary signals:

- abstraction
- representation-invariant reasoning
- transfer robustness

Failure mode exposed:

- solving from lexical familiarity rather than structure

## C. Metacognitive Revision

The model is asked for an initial answer, confidence, and rule hypothesis under ambiguous evidence. It is then shown corrective evidence that resolves the ambiguity.

Primary signals:

- uncertainty awareness
- calibration
- hypothesis revision

Failure mode exposed:

- premature certainty
- failure to update after contradiction

## D. Attention and Distractor Control

Each task mixes relevant and irrelevant structure. Decoy fields are intentionally patterned and can be more salient than the true signal.

Primary signals:

- selective attention
- robustness to salience traps
- resistance to spurious structure

Failure mode exposed:

- following the loudest pattern instead of the generative one

## E. Social Cognition Miniworlds

Small text worlds create asymmetries in observation, belief, and incentives. Agents can hold stale beliefs after a hidden move, and some speakers may try to help while others try to mislead.

Primary signals:

- false-belief reasoning
- source reliability tracking
- incentive-aware social inference

Failure mode exposed:

- treating every confident statement as equally trustworthy

## Shared Schema

Each task record uses a common schema:

- `task_id`
- `family`
- `difficulty`
- `context`
- `examples`
- `query`
- `answer`
- `metadata`
- `latent_rule_summary`
- `shift_type`
- `distractor_level`
- `scoring_notes`

This keeps family-specific task logic compatible with a common export and evaluation pipeline.

## Metrics

AGUS v1 includes the following model-agnostic metrics:

### Accuracy

Exact-match score over scorable outputs.

### Adaptation Speed

For hidden-rule tasks, measure how quickly predictions stabilize after the shift.

### Transfer Score

For shift-transfer tasks, score whether the latent rule survives remapping of surface symbols.

### Calibration Score

For metacognitive tasks, compare predicted confidence to expected certainty before disambiguation and to correctness after revision.

### Revision Quality

Measure whether the solver actually changes the answer when it should, lands on the revised target, and updates confidence appropriately.

### Distractor Robustness

Measure degradation in attention tasks as distractor load increases.

## AGUS Interactive

AGUS Interactive adds a lightweight turn-based protocol on top of the static task suite. Rather than collecting only one final answer, the benchmark can now ask a model to:

1. infer a hypothesis from initial evidence
2. report confidence
3. receive a contradiction, correction, or representation shift
4. revise both the answer and the hypothesis
5. continue through a short 3-5 turn micro-episode when attention or social belief tracking is central to the task

This protocol currently supports:

- hidden rule induction
- shift and transfer
- metacognitive revision
- attention and distractor control
- social cognition miniworlds

### Why AGUS Interactive Is More Novel Than Static Reasoning Datasets

Most reasoning datasets are static snapshots. They do not directly measure whether a model can update its beliefs once the world changes or once new evidence arrives. AGUS Interactive is more novel because it scores the **trajectory** of reasoning rather than only the endpoint.

The interactive layer measures:

- online hypothesis revision
- contradiction sensitivity
- confidence recalibration
- adaptation gain after evidence
- belief trajectory quality over time
- attention recovery after distractor capture
- trust revision after deceptive evidence
- belief-state consistency across multiple turns

This is closer to the cognitive abilities the Kaggle framing cares about: flexibility, self-monitoring, and adaptive generalization under shift.

### Why 3-5 Turn Micro-Episodes Matter

Two-turn revision is already stronger than one-shot evaluation, but it still leaves room for the critique that the benchmark is just "answer once, then answer again." The 3-5 turn episodes for attention and social cognition reduce that criticism because they require a solver to maintain a coherent internal state across multiple updates.

In AGUS Interactive v2, micro-episodes let us measure:

- selective attention recovery over time rather than in one jump
- whether distractors capture the model before a cue arrives
- whether confidence drops under clutter and rises after disambiguation
- whether trust shifts after deceptive or incentive-revealing evidence
- whether world state and agent belief state remain separated over multiple turns

### Remaining Shortcut Risks

Interactive evaluation still has shortcut risks. A model might learn to emit generic phrases like "I revise my hypothesis" without genuinely tracking the latent structure. It might also exploit shallow regularities in the evidence templates.

AGUS mitigates these risks by:

- scoring answer improvement, not just self-reported revision
- checking contradiction acknowledgement only when contradiction is actually present
- pairing confidence movement with correctness-sensitive recalibration
- separating rule-changing tasks from rule-preserving transfer tasks
- separating attention recovery from rule revision and separating trust revision from world-state tracking
- using deterministic, inspectable evidence injection instead of unconstrained dialogue

## Adversarial Curation

Synthetic task generation is necessary for breadth, but by itself it is not enough for benchmark legitimacy. A synthetic suite can still be too templated, too easy, or too friendly to shallow shortcuts. AGUS therefore adds a deterministic adversarial curation layer on top of generation.

The curation layer runs lightweight baseline probes that approximate shortcut behavior:

- surface similarity matching
- local majority-rule overgeneralization
- distractor-vulnerable salience following
- no-revision behavior after contradiction
- trust-naive social reasoning
- representation anchoring under transfer

For each task or interactive episode, the curation layer estimates:

- `baseline_solve_rate`
- `shortcut_vulnerability_score`
- `revision_discrimination_score`
- `distractor_discrimination_score`
- `social_reasoning_discrimination_score`
- `transfer_depth_score`
- `trajectory_value_score`
- `benchmark_signal_score`

This gives AGUS a second stage after generation:

1. generate candidate tasks broadly
2. challenge them with weak but revealing heuristic probes
3. retain the ones that better separate adaptive cognition from shortcut behavior

### Why Adversarial Curation Improves Benchmark Legitimacy

This helps answer a common criticism of synthetic benchmarks: that they are too easy to game once the template family is recognized. AGUS Curation does not solve that criticism completely, but it does reduce it in a concrete way by removing or downranking tasks that weak heuristic agents can already solve.

The key legitimacy gain is not that curated tasks become impossibly hard. The gain is that retained tasks are more likely to be:

- less shortcut-solvable
- more revision-sensitive
- more trajectory-dependent
- more discriminative of adaptive cognition

## Adversarial Refinement Loop

AGUS now treats curation as a generator-improvement signal rather than a terminal filter. The refinement loop inspects rejected tasks and weak-solver success patterns, then uses those signals to harden the weakest generators before the next curation pass.

In v1, the main refinement targets are:

- `attention_distractors`
  - diversify distractor mechanisms
  - delay the causal cue
  - choose query rows that punish nearest-example matching
- `shift_transfer`
  - compose representation changes
  - vary remap profiles
  - add bridge representations that require maintaining the rule while revising the codebook hypothesis

This creates a more defensible benchmark development loop:

1. generate candidate tasks
2. run weak heuristic probes
3. analyze rejection patterns and shortcut wins
4. modify the weakest generators
5. regenerate and compare pre/post curation outcomes

### Why This Helps Benchmark Legitimacy

One common criticism of synthetic benchmarks is that they are static artifacts produced by a single design pass. The refinement loop reduces that criticism by showing that AGUS is explicitly adversarial against its own shortcut vulnerabilities.

The benchmark is therefore not just:

- generated once
- lightly filtered
- submitted as-is

It is iteratively hardened using observed weak-solver failures, which makes its retained tasks more discriminative and its design process more transparent.

## Probe-Conditioned Generator Search

AGUS now adds a third layer after generation and curation: generator search. Rather than assuming that one manually chosen generator setting is best, the benchmark can now evaluate multiple compact, deterministic generator configurations and rank them by diagnostic value.

This search is not model hyperparameter tuning. It is benchmark-construction search. Candidate generator settings are scored by how they behave under adversarial curation, with emphasis on:

- kept rate after curation
- average benchmark signal
- shortcut resistance
- trajectory value
- family-specific discrimination

The current search focuses on the two families where generator design matters most for discriminative validity:

- `attention_distractors`
  - distractor diversity
  - cue delay
  - anti-template strength
  - adversarial query selection mode
- `shift_transfer`
  - remap composition depth
  - bridge representation mode
  - anti-anchor strength
  - latent rule mix

### Why This Improves Defensibility

Probe-conditioned search makes AGUS more systematic in a way that should be legible to both Kaggle judges and researchers. The benchmark is no longer just:

1. generated
2. filtered
3. lightly hand-tuned

It is now:

1. generated under several explicit candidate designs
2. challenged by weak heuristic probes
3. ranked by discriminative signal and shortcut resistance
4. updated using the strongest generator settings

That makes the benchmark-development loop more transparent, more reproducible, and less dependent on informal taste.

## Search-Conditioned Refinement

AGUS now supports a search-conditioned refinement mode that promotes winning generator configurations directly into the next refinement pass. This means the benchmark-development loop is no longer:

1. generate
2. curate
3. manually refine

It can now be:

1. generate
2. curate
3. search generator settings under weak-solver pressure
4. promote the winning configurations
5. regenerate and re-curate
6. compare the result against the prior manual refinement baseline

The point is not to optimize for one narrow leaderboard behavior. The point is to make benchmark construction itself more adaptive and more evidence-driven.

### Why This Strengthens The Benchmark Story

For judges and researchers, this matters because AGUS can now show a fully closed-loop development process:

- weak baselines expose failure modes
- search explores generator settings that increase diagnostic pressure
- refinement adopts the winning settings
- the benchmark reports whether this actually improves retained signal relative to prior manual refinement

That makes AGUS easier to defend as a benchmark-development methodology rather than only as a static benchmark artifact.

## Local-First Model Evaluation

AGUS evaluation is intentionally local-first. Under realistic research budgets, the benchmark needs to remain runnable without paid frontier APIs, so the evaluation harness supports deterministic mock adapters, local Ollama-style inference, resumable run directories, and structured failure extraction.

This matters for benchmark legitimacy because it separates two questions:

1. is the benchmark construction pipeline sound and reproducible?
2. how do particular models perform on it?

The current harness lets us make strong claims about the first question even before running large frontier-model sweeps. It also makes AGUS practical on a single workstation by supporting overnight runs, incremental resume, and artifact-level inspection of where a model failed statically or during interactive revision.

## Why This Feels Different

AGUS is not just a bundle of static logic questions. Its novelty comes from **sequencing cognitive pressures**:

- learn a rule from little data
- face a shift
- transfer across representations
- expose confidence
- revise after contradiction
- ignore attractive but irrelevant information
- reason about beliefs and intentions under partial observability

That sequencing makes the benchmark more diagnostic of online cognitive competence than standard static test collections.
