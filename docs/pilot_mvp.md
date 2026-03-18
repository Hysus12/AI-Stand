# Pilot MVP

## What This Version Does

This Pilot MVP turns the validated `deepseek_direct + heuristic fallback` route into a
project-level synthetic survey engine for early-stage product decisions.

Current scope:

- 7-15 closed-ended questions
- 1-3 target segments, with 2-3 as the expected operating pattern
- 0-3 variants
- Question-level distributions, segment aggregates, synthetic respondents, summary, recommendations, and export artifacts

## What It Does Not Do

- Open-ended qualitative synthesis at scale
- Skip logic / routing-heavy surveys
- Long enterprise questionnaires
- Political, medical, or other high-risk policy use cases
- Superiority claims over human-run surveys

## How To Run

```bash
spbce-mvp run-project D:\dev\Gnosis\configs\pilot_mvp_sample.yaml --output-dir D:\dev\Gnosis\reports\pilot_mvp_sample_output
```

Optional:

- `--synthetic-count 1000` to override the respondent count in the project spec
- `--env-file path\to\.env` to load DeepSeek credentials

To compare a completed run against a public reference file:

```bash
spbce-mvp compare-project D:\dev\Gnosis\configs\samples\sample_en_pew_policy_topline.yaml D:\dev\Gnosis\reports\sample_runs\sample_en_pew_policy_topline\project_result.json D:\dev\Gnosis\data\reference_results\sample_en_pew_policy_topline_reference.json --output-dir D:\dev\Gnosis\reports\sample_runs\sample_en_pew_policy_topline
```

## Artifacts

The CLI writes:

- `project_result.json`: complete structured project output
- `synthetic_respondents.csv`: flat respondent table for spreadsheet / BI use
- `synthetic_respondents.jsonl`: row-wise export
- `executive_summary.md`: summary for slides or memos
- `segment_report.json`: per-segment and per-variant output
- `recommendations.json`: structured recommendations
- `question_level_results.json`: cross-segment question diagnostics

## Notes

- Primary route is `deepseek_direct`; invalid, timeout, schema-failure, or low-quality outputs fall back to `heuristic`.
- Synthetic rows use a lightweight latent-profile sampler to reduce row-level independence noise, but they are not a learned joint respondent model.
- Behavior validity remains an explicit limitation until real customer paired data exists.
- Public sample-project notes live in `docs/sample_projects_public.md`.
