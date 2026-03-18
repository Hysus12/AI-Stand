# Executive Summary: TEDS2018 政二代態度樣本

## Snapshot
- Primary route: `deepseek_direct -> heuristic fallback` with 0.0% fallback usage.
- Most promising segment: **地方務實型選民**. 地方務實型選民 has the strongest aggregate score (0.532) with mean confidence 0.444.
- Mean confidence: 0.490; mean uncertainty: 0.510.

## Segment And Question Signals
- Largest divergence: **q09_democracy_problem** (avg JS 0.046).
- Most stable conclusions: q15b_effectiveness (0.65), q15c_resources (0.64), q15a_corruption (0.68).
- Least stable conclusions: q09_democracy_problem (0.33), q14_democratic_expression (0.35), q12_public_impression (0.44).

## Recommended Next Actions
- Prioritize 地方務實型選民: 地方務實型選民 has the strongest aggregate score (0.532) with mean confidence 0.444.
- Use q09_democracy_problem as a segmentation lever: 下面哪一個最像是我國民主制度最嚴重的問題？ shows the largest cross-segment divergence (avg JS 0.046).
- Treat close calls as validation candidates: Higher-uncertainty or unstable questions should be converted into focused follow-up tests before committing large spend.

## Risks And Limits
- Pilot MVP only supports 7 to 15 closed-ended questions and up to 3 segments / 3 variants.
- Synthetic respondents use a lightweight latent-profile sampler rather than a learned joint respondent model.
- Behavior validity has not been validated on customer-specific commercial outcomes; do not claim superiority over human surveys.
- Routing / skip logic and long open-ended questionnaires are intentionally out of scope.
