# DeepSeek v3 Product Route Summary

- Recommendation: `A`
- Stop combiner research: `true`
- Overall heuristic JS: `0.0711`
- Overall DeepSeek JS: `0.0245`
- DeepSeek fallback rate: `0.0080`
- DeepSeek JSON compliance rate: `0.9920`

## Product recommendation
- `deepseek_direct` should be the MVP main model.
- `heuristic` should remain the fallback path.
- Combiner research can be paused for now.

## Productization checklist
- add auth and per-key rate limiting
- add persistent request logging and tracing
- add monitoring for fallback rate, invalid rate, latency, and cost
- add dashboarding for token usage and cost by tenant
- add API docs and operational runbooks
