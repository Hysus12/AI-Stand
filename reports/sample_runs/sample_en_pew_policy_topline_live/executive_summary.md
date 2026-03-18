# Executive Summary: Pew Debt Reduction Policy Sample

## Snapshot
- Primary route: `deepseek_direct -> heuristic fallback` with 0.0% fallback usage.
- Most promising segment: **Social program defenders**. Social program defenders has the strongest aggregate score (0.611) with mean confidence 0.784.
- Mean confidence: 0.684; mean uncertainty: 0.316.

## Segment And Question Signals
- Largest divergence: **q35j_student_loans** (avg JS 0.097).
- Most stable conclusions: q35m_high_income_tax (0.69), q35d_mortgage_deduction (0.62), q35h_defense_cuts (0.68).

## Recommended Next Actions
- Prioritize Social program defenders: Social program defenders has the strongest aggregate score (0.611) with mean confidence 0.784.
- Use q35j_student_loans as a segmentation lever: Reduce federal funding for college student loan programs. shows the largest cross-segment divergence (avg JS 0.097).

## Risks And Limits
- Pilot MVP only supports 7 to 15 closed-ended questions and up to 3 segments / 3 variants.
- Synthetic respondents use a lightweight latent-profile sampler rather than a learned joint respondent model.
- Behavior validity has not been validated on customer-specific commercial outcomes; do not claim superiority over human surveys.
- Routing / skip logic and long open-ended questionnaires are intentionally out of scope.
