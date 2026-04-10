# Final Requirements Audit

## Definitely Ready

- AGUS benchmark repo with generation, interactive evaluation, adversarial curation, refinement, search, local evaluation, failure distillation, and instability analysis
- Judge-facing results docs and Learning-track framing
- Kaggle benchmark packaging bundle in `kaggle_benchmark/` for the mandatory benchmark attachment requirement
- Clean local empirical result story with named artifacts
- Clear headline supported by current evidence:
  - Llama higher static accuracy
  - Qwen higher adaptive reasoning quality
  - the Llama-versus-Qwen split holds on `3/3` deterministic replication slices
  - AGUS reveals the divergence
- Failure examples that are legible and interpretable

## Likely Ready

- Learning-track positioning is strong and directly aligned with the official framing around learning and adaptive evaluation
- Supporting use of metacognition, attention, executive flexibility, and social cognition is credible
- The current README and docs now support a submission-oriented first impression

## Manual Kaggle-Side Steps Still Required

- Fill in final team name
- Fill in final organizational affiliation
- Create the Kaggle benchmark notebook from the benchmark entrypoint
- Run and publish the AGUS benchmark package in Kaggle
- Copy the published Kaggle benchmark project link
- Paste or adapt the final writeup into Kaggle’s submission form
- Attach the correct public repo link
- Attach the Kaggle benchmark project link to the writeup
- Add any optional cover image
- Add any optional public notebook link if desired
- Perform the final UI review inside Kaggle before submission

## Uncertain Or Must Not Be Hallucinated

- Exact Kaggle submission form wording, field limits, and attachment affordances should be verified manually in the competition UI
- Whether a cover image is required, optional, or unused must be checked manually
- Whether a public notebook materially helps judging in this competition should be decided manually
- Any official team-affiliation language must come from the submitter, not from this repo

## Submission-Readiness Verdict

The repository is ready for a final packaging commit from a documentation, evidence, and benchmark-packaging standpoint. The remaining steps are mainly Kaggle-side publication and submission actions rather than repo-development blockers.
