# Executive Summary: Pew Debt Reduction Policy Sample

## Snapshot
- Primary route: `deepseek_direct -> heuristic fallback` with 100.0% fallback usage.
- Most promising segment: **Fiscal hawks**. Fiscal hawks has the strongest aggregate score (0.491) with mean confidence 0.431.
- Mean confidence: 0.431; mean uncertainty: 0.569.

## Segment And Question Signals
- Largest divergence: **q35a_low_income_programs** (avg JS 0.000).
- Least stable conclusions: q35a_low_income_programs (0.43), q35b_social_security_age (0.43), q35c_education_cuts (0.43).

## Recommended Next Actions
- Prioritize Fiscal hawks: Fiscal hawks has the strongest aggregate score (0.491) with mean confidence 0.431.
- Use q35a_low_income_programs as a segmentation lever: Reduce federal funding for programs that help lower income Americans. shows the largest cross-segment divergence (avg JS 0.000).
- Track route reliability in pilot delivery: Fallbacks were triggered in 100.0% of question predictions, so project logs and retry monitoring should stay on.
- Treat close calls as validation candidates: Higher-uncertainty or unstable questions should be converted into focused follow-up tests before committing large spend.

## Risks And Limits
- Pilot MVP only supports 7 to 15 closed-ended questions and up to 3 segments / 3 variants.
- Synthetic respondents use a lightweight latent-profile sampler rather than a learned joint respondent model.
- Behavior validity has not been validated on customer-specific commercial outcomes; do not claim superiority over human surveys.
- Routing / skip logic and long open-ended questionnaires are intentionally out of scope.
