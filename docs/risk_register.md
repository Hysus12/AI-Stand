# SPBCE Risk Register

## R1. Dataset mismatch to target product

Description:
The initial public survey dataset is useful for pipeline bootstrapping but does not fully match customer concept tests, campaign surveys, or product launches.

Impact:
High.

Mitigation:
Keep the schema general, make the behavior layer pluggable, and document that first results are survey-fidelity bootstrapping only.

## R2. Group leakage in evaluation

Description:
If rows from the same question or study leak across splits, benchmark performance will be inflated and unusable.

Impact:
High.

Mitigation:
Default to group-aware splitting, add split validation tests, and fail builds when overlapping groups are detected.

## R3. Prompt-only baseline mis-specification

Description:
A weak prompt-only baseline can make the supervised model look better than it is.

Impact:
Medium to high.

Mitigation:
Implement a transparent zero-shot baseline now and reserve room for a stronger API-backed prompt benchmark later.

## R4. Calibration overfit

Description:
Calibration fit on the wrong partition will overstate reliability and understate uncertainty.

Impact:
High.

Mitigation:
Fit calibration on validation predictions only, never on test.

## R5. Behavior model unsupported by public data

Description:
Public paired survey-behavior data is scarce, so behavior calibration may remain mostly architectural until private data is available.

Impact:
High.

Mitigation:
Ship a stable interface, mock-compatible schema, and clear benchmark plan without overclaiming current behavior validity.

## R6. Unsupported population extrapolation

Description:
The model may be queried on populations with weak or no training coverage.

Impact:
High.

Mitigation:
Implement novelty-based OOD heuristics and force explicit support notes in responses.

## R7. License and usage restrictions

Description:
Some public datasets are restricted, non-commercial, or registration-gated.

Impact:
High.

Mitigation:
Track licenses in dataset inventory and cards. Avoid treating bootstrap datasets as deployment-approved training corpora.

## R8. Synthetic respondent misuse

Description:
Users may mistake sampled synthetic respondents for observed people.

Impact:
Medium.

Mitigation:
Label all synthetic outputs clearly and ensure the API returns distributions as the primary object.

## R9. Overclaiming survey-to-behavior mapping

Description:
Intent is not action. A model that predicts survey responses well may still predict real outcomes poorly.

Impact:
High.

Mitigation:
Keep survey fidelity and behavior validity as separate benchmark tracks and require held-out behavior evidence before claims.

## R10. Operational fragility

Description:
Heavy baseline dependencies and external model downloads can make the toy pipeline brittle.

Impact:
Medium.

Mitigation:
Provide seeded demo artifacts, modular scripts, and deterministic offline fallbacks where possible.

## What was implemented

- first-pass project risk register tied to data, evaluation, product, and operations

## Assumptions made

- the initial repo should optimize for honesty and reproducibility over breadth

## Risks and unknowns

- paired data availability remains the main strategic blocker

## Not implemented yet

- issue owners
- probability/severity scoring framework
- mitigation tracking dashboard

## Exact next step

Wire the highest-severity mitigations directly into code: schema validation, split validation, OOD notes, and non-commercial dataset documentation.
