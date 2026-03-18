# Initial Success Criteria

## First milestone pass criteria

- one real public survey-style dataset ingested end to end
- canonical schema validation passing on normalized records
- leakage-safe group-aware split generation
- prompt-only baseline runnable
- simple supervised survey prior baseline runnable
- held-out survey fidelity metrics generated from the same pipeline
- minimal `/predict-survey` and `/sample-respondents` endpoints available

## Second milestone pass criteria

- calibration layer improves at least one core survey-fidelity metric on validation without harming held-out robustness materially
- uncertainty bucket correlates with held-out error
- OOD flag is elevated on known unsupported cases
- behavior model interface accepts paired records cleanly

## Third milestone pass criteria

- paired survey-behavior data available
- AI-only and hybrid models are benchmarked against a human-survey-only baseline on held-out groups
- claims are limited to measured deployment settings
