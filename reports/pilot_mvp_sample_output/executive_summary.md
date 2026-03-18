# Executive Summary: NimbusNotes

## Snapshot
- Primary route: `deepseek_direct -> heuristic fallback` with 6.2% fallback usage.
- Most promising segment: **Startup ops leads**. Startup ops leads has the strongest aggregate score (0.676) with mean confidence 0.426.
- Best variant to lead with: **Pro monthly (v_pro)**. Pro monthly (v_pro) leads on average weighted score (0.637) across segments.
- Mean confidence: 0.384; mean uncertainty: 0.616.

## Segment And Question Signals
- Largest divergence: **q_interest** (avg JS 0.028).
- Least stable conclusions: q_interest (0.36), q_value (0.36), q_price (0.36).

## Recommended Next Actions
- Prioritize Startup ops leads: Startup ops leads has the strongest aggregate score (0.676) with mean confidence 0.426.
- Lead with Pro monthly (v_pro): Pro monthly (v_pro) leads on average weighted score (0.637) across segments.
- Use q_interest as a segmentation lever: How likely would you be to try this product in the next 30 days? shows the largest cross-segment divergence (avg JS 0.028).
- Track route reliability in pilot delivery: Fallbacks were triggered in 6.2% of question predictions, so project logs and retry monitoring should stay on.

## Risks And Limits
- Pilot MVP only supports 7 to 15 closed-ended questions and up to 3 segments / 3 variants.
- Synthetic respondents use a lightweight latent-profile sampler rather than a learned joint respondent model.
- Behavior validity has not been validated on customer-specific commercial outcomes; do not claim superiority over human surveys.
- Routing / skip logic and long open-ended questionnaires are intentionally out of scope.
