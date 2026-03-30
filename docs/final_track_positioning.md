# Final Track Positioning

## Why Learning Is The Strongest Fit

AGUS is best positioned as a **Learning-track** submission because its core mechanism is adaptation under sparse evidence and subsequent shift. The benchmark repeatedly asks whether a model can:

- infer a latent rule from minimal data
- detect that the old rule no longer applies
- revise its working hypothesis
- transfer the rule into a changed representation

That is closer to learning-as-adaptation than to one-shot reasoning or pure static accuracy.

## Why Metacognition, Attention, And Social Cognition Are Supporting

These modules remain important, but they strengthen the Learning claim rather than replace it.

- **Metacognition** supports the benchmark by measuring whether the model knows when it is uncertain and revises after correction.
- **Attention** supports the benchmark by testing whether learning survives distractors and whether the model can recover when relevant cues appear late.
- **Social cognition** supports the benchmark by testing whether updated beliefs remain coherent when evidence is distributed across agents with different incentives and knowledge states.

In other words, these modules help show that AGUS is measuring *adaptive learning quality*, not just narrow symbolic rule fitting.

## How AGUS Maps To The Competition Framing

The official Google DeepMind framing describes the Kaggle hackathon as targeting five cognitive abilities where the evaluation gap is largest: learning, metacognition, attention, executive functions, and social cognition. AGUS maps to that framing as follows:

- **Learning**: primary emphasis through hidden-rule induction, rule-shift adaptation, and transfer under remapping
- **Metacognition**: confidence reporting, contradiction handling, and rule revision
- **Attention**: distractor resistance, cue utilization, and attention recovery
- **Executive functions**: cognitive flexibility under rule changes and multi-turn adaptation
- **Social cognition**: belief tracking, trust revision, and deceptive-evidence handling

The strongest submission story is therefore:

**AGUS is a Learning-track benchmark whose supporting modules make the learning signal more credible and more diagnostic.**

## Recommended Judge-Facing Position

Lead with:

- adaptive generalization under shift
- static-versus-dynamic divergence
- interactive revision and instability evidence

Use metacognition, attention, and social cognition as evidence that AGUS is measuring a richer and more defensible form of learning than a simple rule-induction dataset would.
