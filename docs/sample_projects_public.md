# Public Sample Projects

## Purpose

These samples are not benchmark research. They are end-to-end product fixtures that test whether
the Pilot MVP can:

- ingest a real public questionnaire,
- run project-level inference,
- export deliverable artifacts,
- and compare AI output with public reported results without claiming superiority.

## Chinese Sample

- Dataset: `TEDS2018_SE01` political-family survey experiment
- Questionnaire source: SRDA public questionnaire PDF
- Public result source: public paper and table values based on the same survey
- Why it fits:
  - attitude / political-family / fairness / corruption / scenario framing
  - closed-ended items
  - no product-usage or satisfaction dependency

## English Sample

- Dataset: Pew Research Center, Early October 2012 Political Survey topline
- Question block: Q35 debt-reduction proposals
- Public result source: Pew topline PDF
- Why it fits:
  - approve / disapprove policy items
  - stable closed-ended structure
  - direct question-level public percentages

## Comparison Rules

- Public results are stored separately in `data/reference_results`.
- The sample configs do not contain public distributions.
- Comparison happens only after a project run completes.
- Some Chinese references are approximate because the public paper reports topline prose rather than full microdata tables.

## Limits

- The Chinese sample is manually normalized from the public questionnaire and paper, not from an automated parser.
- Several Chinese questions are intentionally left unmatched because no clean public topline was available.
- These samples are for product demonstration and plumbing validation, not for external accuracy claims.
