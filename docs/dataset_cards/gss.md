# Dataset Card: General Social Survey Public-Use Cumulative Data

## Source

- NORC General Social Survey public-use data
- Download path used in this repo: `https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/GSS_stata.zip`

## Intended role in SPBCE

GSS is the first stronger public microdata survey source in the repo. It improves subgroup realism, enables temporal evaluation, and supports a grouped survey-behavior proxy benchmark.

## Access and license

- Public-use download from NORC
- See GSS data access terms and citation guidance on the NORC site

## Coverage strengths

- respondent-level microdata
- long time span
- demographic subgroup fields
- stable repeated-attitude questions
- multiple self-reported behavior variables

## Canonical survey mapping in this repo

- demographic grouping: `year`, `sex`, `degree`, `age_band`
- survey variables used by default:
  - `trust`
  - `helpful`
  - `fair`
  - `partyid`
  - `polviews`
  - `natheal`
  - `natcrime`
  - `natenvir`
- weighting: `wtssall`

Each grouped `(question, year, subgroup)` aggregate becomes a `SurveyRecord`.

## Paired behavior proxy mapping in this repo

The current public paired benchmark is a grouped proxy, not a market-outcome dataset.

- survey inputs: grouped attitude distributions
- outcome variables:
  - `boycott`
  - `signpet`
  - `conoffcl`
  - `polfunds`
- target: weighted subgroup behavior rate

Each grouped `(year, subgroup, outcome)` aggregate becomes a `PairedSurveyBehaviorRecord`.

## Known limitations

- outcomes are self-reported survey behaviors, not logged external market outcomes
- proxy benchmark years are limited by outcome-variable availability
- all records are US-only
- stimulus realism is weaker than customer product/campaign data

## SPBCE usage notes

GSS is strong enough to harden survey-fidelity evaluation and to exercise the behavior-validity pipeline. It is not sufficient for claims about campaign CTR, purchase conversion, product adoption, or event attendance.
