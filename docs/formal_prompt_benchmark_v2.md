# Formal Prompt Benchmark v2 Candidate

- Manifest: `data/processed/gss_survey_splits/formal_prompt_holdout_v2_100.json`
- Audit: `reports/audits/formal_prompt_holdout_v2_100_audit.json`
- Status: `pass`
- Requested sample size: `100`
- Selected records: `100`
- Held-out question_ids: `partyid, polviews, natcrime`
- Excluded question_ids: `fair, helpful, natenvir, natheal, trust`

## Expansion vs formal50 v1

- formal50 v1 held out `fair` and `helpful` only.
- v2 candidate expands to `partyid`, `polviews`, and `natcrime`.
- v2 candidate increases frozen test size from `50` to `100`.
- Anti-leakage validation still reports zero overlap on held-out `question_id` and normalized question text.

## Audit highlights

- Few-shot pool size: `1738`
- Excluded records from few-shot pool: `0`
- Test question counts: `{'natcrime': 23, 'partyid': 41, 'polviews': 36}`
- Year span: `1972` to `2024`

## Limitation

- This is a frozen v2 candidate and audit-ready skeleton. I did not run a full v2 multi-model benchmark in this pass.