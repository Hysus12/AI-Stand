# Initial Dataset Inventory

This inventory is intentionally split into two buckets: survey-prior datasets for response distribution learning, and paired survey-behavior candidates for downstream outcome calibration.

## Survey-prior candidates

| Dataset | Source | Access / license notes | Subgroup coverage | Format | Expected preprocessing burden | Fit for SPBCE |
| --- | --- | --- | --- | --- | --- | --- |
| Anthropic `llm_global_opinions` | Hugging Face, derived from WVS and Pew Global Attitudes | CC-BY-NC-SA-4.0, non-commercial, public download | Country only in released format | Question text + options + per-country distributions | Low | Good bootstrap toy dataset |
| General Social Survey (GSS) | NORC | Public-use data available via GSS Data Explorer; some sensitive variables withheld | Rich US demographics | Microdata and extracts | Medium | Strong US survey prior candidate |
| European Social Survey (ESS) | ESS ERIC | Freely accessible with portal access/registration workflows | Rich demographics across Europe | Microdata | Medium | Strong cross-national prior candidate |
| World Values Survey (WVS) | WVS Association | Public data access with project documentation and registration workflows | Country and respondent covariates | Microdata | Medium | Strong values/opinion prior candidate |
| European Values Study (EVS) | GESIS / EVS | Registration required for data downloads | Rich demographics | Microdata | Medium | Good Europe-focused supplement |
| ANES Time Series / Cumulative | ANES | Public access via Data Center | Rich US political demographics | Microdata | Medium | Strong political-opinion benchmark |
| Afrobarometer | Afrobarometer | Free public data with data-use policy; some access gating for certain files | Country, region, respondent covariates | Country and merged microdata | Medium | Strong Africa coverage |
| Pew Research Center public datasets | Pew Research Center | Account registration and terms required | Often rich demographics | Microdata / toplines | Medium | Good thematic supplement |
| Cooperative Election Study (CES/CCES) | Harvard Dataverse | Public dataverse distribution | Rich US political and demographic fields | Microdata | Medium | Strong US election benchmark |
| US World Values Survey | US-WVS | Public download portal | US demographics | Microdata | Low to medium | Useful US-specific supplement |

## Paired survey-behavior candidates

| Dataset | Public status | Outcome fields | Current suitability |
| --- | --- | --- | --- |
| Campaign lift or ad-study datasets | Rare publicly | CTR, CVR, lift | Usually private; best handled through pluggable private schema |
| Product concept test plus launch outcomes | Rare publicly | Purchase, adoption, sales proxy | Usually private; primary target for customer integrations |
| Event intention plus attendance logs | Rare publicly | Registration, attendance | Good internal/private data target |
| Survey plus digital behavior panels | Mixed / restricted | Clicks, subscriptions, retention | Often privacy-restricted |

## Recommended initial path

1. Use `llm_global_opinions` to validate survey-fidelity plumbing.
2. Add one richer microdata survey dataset next, likely GSS, ESS, or ANES depending access friction.
3. Use GSS as the first implemented stronger microdata source.
4. Treat behavior calibration as a pluggable interface until paired customer data or a reliable public benchmark is secured.

## Source notes

- [Anthropic `llm_global_opinions`](https://huggingface.co/datasets/Anthropic/llm_global_opinions)
- [GSS data access](https://gss.norc.org/get-the-data.html)
- [ESS data portal](https://www.europeansocialsurvey.org/data-portal)
- [Afrobarometer data access](https://www.afrobarometer.org/data/)
- [Pew data access](https://www.pewresearch.org/short-reads/2021/10/22/how-to-access-pew-research-center-survey-data/)
- [CCES data portal](https://cces.gov.harvard.edu/data)
