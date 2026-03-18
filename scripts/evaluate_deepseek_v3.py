from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from spbce.baselines.prompt_only import PromptOnlyPersonaBaseline
from spbce.data.datasets import load_survey_records
from spbce.inference.mvp import MvpInferenceEngine
from spbce.metrics.distributions import (
    js_divergence,
    probability_mae,
    probability_rmse,
    safe_kl_divergence,
    top_option_accuracy,
)
from spbce.schema.api import PredictSurveyRequest, SurveyContext
from spbce.schema.canonical import SurveyRecord
from spbce.settings import get_provider_environment_summary, initialize_runtime_env
from spbce.utils.io import read_json, write_json
from spbce.utils.prompt_benchmark import (
    derive_contaminated_question_blacklist,
    normalize_question_text,
    question_ids_from_formal_manifest,
    validate_formal_holdout,
)
from spbce.utils.text import simple_tokenize


def filter_records(records: list[SurveyRecord], record_ids: list[str]) -> list[SurveyRecord]:
    allowed = set(record_ids)
    return [record for record in records if record.record_id in allowed]


def year_bucket(record: SurveyRecord) -> str:
    year = int(record.wave_id or "0")
    if year <= 0:
        return "unknown"
    return f"{(year // 10) * 10}s"


def question_length_bucket(record: SurveyRecord) -> str:
    token_count = len(simple_tokenize(record.question_text))
    if token_count <= 3:
        return "short_1_3"
    if token_count <= 6:
        return "medium_4_6"
    return "long_7_plus"


def population_bucket(record: SurveyRecord) -> str:
    struct = record.population_struct
    parts = [
        struct.age_band or "age_any",
        struct.gender or "gender_any",
        struct.education or "edu_any",
    ]
    return "|".join(parts)


def metric_summary(rows: list[dict[str, Any]], predictor_key: str) -> dict[str, float]:
    if not rows:
        return {
            "js_divergence": 0.0,
            "safe_kl_divergence": 0.0,
            "probability_mae": 0.0,
            "probability_rmse": 0.0,
            "top_option_accuracy": 0.0,
        }
    return {
        "js_divergence": float(sum(row[predictor_key]["js"] for row in rows) / len(rows)),
        "safe_kl_divergence": float(sum(row[predictor_key]["kl"] for row in rows) / len(rows)),
        "probability_mae": float(sum(row[predictor_key]["mae"] for row in rows) / len(rows)),
        "probability_rmse": float(sum(row[predictor_key]["rmse"] for row in rows) / len(rows)),
        "top_option_accuracy": float(
            sum(row[predictor_key]["top1"] for row in rows) / len(rows)
        ),
    }


def deepseek_operational_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "fallback_rate": 0.0,
            "invalid_output_rate": 0.0,
            "json_compliance_rate": 0.0,
            "served_invalid_flag_rate": 0.0,
            "total_input_tokens": 0.0,
            "total_output_tokens": 0.0,
            "estimated_api_cost_usd": 0.0,
            "total_latency_ms": 0.0,
            "average_latency_ms_per_request": 0.0,
            "total_retry_count": 0.0,
        }
    total_requests = sum(
        row["deepseek"]["cost_latency_summary"]["request_count"] for row in rows
    )
    total_latency_ms = sum(
        row["deepseek"]["cost_latency_summary"]["total_latency_ms"] for row in rows
    )
    return {
        "fallback_rate": float(
            sum(row["deepseek"]["fallback_happened"] for row in rows) / len(rows)
        ),
        "invalid_output_rate": float(
            sum(row["deepseek"]["attempt_invalid"] for row in rows) / len(rows)
        ),
        "json_compliance_rate": float(
            sum(row["deepseek"]["json_success"] for row in rows) / len(rows)
        ),
        "served_invalid_flag_rate": float(
            sum(row["deepseek"]["invalid_flag"] for row in rows) / len(rows)
        ),
        "total_input_tokens": float(
            sum(row["deepseek"]["cost_latency_summary"]["total_input_tokens"] for row in rows)
        ),
        "total_output_tokens": float(
            sum(row["deepseek"]["cost_latency_summary"]["total_output_tokens"] for row in rows)
        ),
        "estimated_api_cost_usd": float(
            sum(row["deepseek"]["cost_latency_summary"]["estimated_api_cost_usd"] for row in rows)
        ),
        "total_latency_ms": float(total_latency_ms),
        "average_latency_ms_per_request": float(total_latency_ms / total_requests)
        if total_requests
        else 0.0,
        "total_retry_count": float(
            sum(row["deepseek"]["cost_latency_summary"]["retry_count"] for row in rows)
        ),
    }


def build_slice_report(rows: list[dict[str, Any]], dimension: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["slice_keys"][dimension])].append(row)
    ordered_keys = sorted(grouped, key=lambda key: (-len(grouped[key]), key))
    return [
        {
            "slice_name": dimension,
            "slice_value": key,
            "record_count": len(grouped[key]),
            "heuristic_prompt_only": metric_summary(grouped[key], "heuristic"),
            "llm_direct_option_probabilities": metric_summary(grouped[key], "deepseek"),
            "deepseek_operational": deepseek_operational_summary(grouped[key]),
            "actual_strategy_breakdown": dict(
                Counter(row["deepseek"]["actual_strategy_used"] for row in grouped[key])
            ),
            "fallback_reason_breakdown": dict(
                Counter(
                    row["deepseek"]["fallback_reason"] or "none" for row in grouped[key]
                )
            ),
        }
        for key in ordered_keys
    ]


def recommend_route(
    overall_heuristic: dict[str, float],
    overall_deepseek: dict[str, float],
    slice_report: dict[str, list[dict[str, Any]]],
) -> tuple[str, bool]:
    dominant_slice_wins = 0
    evaluated_slice_groups = 0
    for dimension in ["question_id", "year_bucket", "option_count", "question_length_bucket"]:
        for row in slice_report.get(dimension, []):
            if row["record_count"] < 5:
                continue
            evaluated_slice_groups += 1
            if (
                row["llm_direct_option_probabilities"]["js_divergence"]
                < row["heuristic_prompt_only"]["js_divergence"]
            ):
                dominant_slice_wins += 1
    deepseek_better_overall = (
        overall_deepseek["js_divergence"] < overall_heuristic["js_divergence"]
        and overall_deepseek["probability_mae"] < overall_heuristic["probability_mae"]
        and overall_deepseek["probability_rmse"] < overall_heuristic["probability_rmse"]
    )
    consistent = evaluated_slice_groups > 0 and (
        dominant_slice_wins / evaluated_slice_groups >= 0.6
    )
    if deepseek_better_overall and consistent:
        return "A", True
    if not deepseek_better_overall and dominant_slice_wins / max(evaluated_slice_groups, 1) < 0.4:
        return "B", False
    return "C", False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--formal-manifest", required=True)
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--formal50-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--comparison-output", required=True)
    parser.add_argument("--slice-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--env-file", default="api.env")
    parser.add_argument("--llm-num-samples", type=int, default=4)
    parser.add_argument("--llm-top-p", type=float, default=0.95)
    parser.add_argument("--llm-max-tokens", type=int, default=128)
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-retry-attempts", type=int, default=1)
    args = parser.parse_args()

    initialize_runtime_env(args.env_file)
    env_summary = get_provider_environment_summary(args.env_file)
    records = load_survey_records(args.input)
    formal_manifest = read_json(args.formal_manifest)
    blacklist = sorted(
        set(derive_contaminated_question_blacklist(args.audit_report))
        | set(question_ids_from_formal_manifest(args.formal50_manifest))
    )
    anti_leakage_audit = validate_formal_holdout(records, formal_manifest, blacklist)
    if anti_leakage_audit["status"] != "pass":
        raise RuntimeError("formal v3 anti-leakage validation failed")

    test_records = filter_records(records, formal_manifest["test_record_ids"])
    heuristic = PromptOnlyPersonaBaseline(backend="heuristic")
    engine = MvpInferenceEngine(
        env_file=args.env_file,
        llm_num_samples=args.llm_num_samples,
        llm_top_p=args.llm_top_p,
        llm_max_tokens=args.llm_max_tokens,
        request_timeout_seconds=args.request_timeout_seconds,
        max_retry_attempts=args.max_retry_attempts,
        default_strategy="deepseek_direct",
        fallback_strategy="heuristic",
    )

    per_record_rows: list[dict[str, Any]] = []
    for record in test_records:
        request = PredictSurveyRequest(
            question_text=record.question_text,
            options=record.options,
            population_text=record.population_text,
            population_struct=record.population_struct,
            context=SurveyContext(product_category=record.domain),
        )
        heuristic_distribution = heuristic.predict_proba(request)
        deepseek_response = engine.predict(request, strategy="deepseek_direct")
        deepseek_distribution = [
            float(deepseek_response.distribution[option]) for option in record.options
        ]
        observed = record.observed_distribution
        direct_metadata = deepseek_response.metadata
        attempt_invalid = bool(
            direct_metadata.get("invalid_output_rate", 0.0) > 0.0
            or deepseek_response.fallback_reason
            in {
                "json_schema_validation_failed",
                "invalid_output",
                "parser_failure",
                "empty_output",
                "timeout",
                "provider_error",
                "low_confidence",
            }
        )
        json_success = bool(float(direct_metadata.get("json_compliance_rate", 0.0)) >= 1.0)
        per_record_rows.append(
            {
                "record_id": record.record_id,
                "question_id": record.question_id,
                "question_text": record.question_text,
                "normalized_question_text": normalize_question_text(record.question_text),
                "population_signature": record.population_signature(),
                "population_text": record.population_text,
                "year": int(record.wave_id or "0"),
                "slice_keys": {
                    "question_id": record.question_id,
                    "year_bucket": year_bucket(record),
                    "population_signature_bucket": population_bucket(record),
                    "option_count": f"{len(record.options)}",
                    "question_length_bucket": question_length_bucket(record),
                    "fallback_happened": (
                        "fallback_yes" if deepseek_response.fallback_happened else "fallback_no"
                    ),
                    "strict_json_success": (
                        "json_success" if json_success else "json_failure"
                    ),
                },
                "heuristic": {
                    "distribution": heuristic_distribution,
                    "js": js_divergence(heuristic_distribution, observed),
                    "kl": safe_kl_divergence(heuristic_distribution, observed),
                    "mae": probability_mae(heuristic_distribution, observed),
                    "rmse": probability_rmse(heuristic_distribution, observed),
                    "top1": top_option_accuracy(heuristic_distribution, observed),
                },
                "deepseek": {
                    "distribution": deepseek_distribution,
                    "requested_strategy": deepseek_response.requested_strategy,
                    "actual_strategy_used": deepseek_response.actual_strategy_used,
                    "fallback_happened": deepseek_response.fallback_happened,
                    "fallback_reason": deepseek_response.fallback_reason,
                    "invalid_flag": deepseek_response.invalid_flag,
                    "json_success": json_success,
                    "attempt_invalid": attempt_invalid,
                    "js": js_divergence(deepseek_distribution, observed),
                    "kl": safe_kl_divergence(deepseek_distribution, observed),
                    "mae": probability_mae(deepseek_distribution, observed),
                    "rmse": probability_rmse(deepseek_distribution, observed),
                    "top1": top_option_accuracy(deepseek_distribution, observed),
                    "cost_latency_summary": deepseek_response.cost_latency_summary.model_dump(),
                    "metadata": deepseek_response.metadata,
                },
            }
        )

    heuristic_overall = metric_summary(per_record_rows, "heuristic")
    deepseek_overall = metric_summary(per_record_rows, "deepseek")
    deepseek_ops = deepseek_operational_summary(per_record_rows)
    overall_results = [
        {
            "predictor": "heuristic_prompt_only",
            **heuristic_overall,
            "invalid_output_rate": 0.0,
            "json_compliance_rate": 0.0,
            "fallback_rate": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "estimated_api_cost_usd": 0.0,
            "total_latency_ms": 0.0,
            "average_latency_ms_per_request": 0.0,
            "total_retry_count": 0.0,
        },
        {
            "predictor": "llm_direct_option_probabilities",
            **deepseek_overall,
            **deepseek_ops,
        },
    ]

    slice_report = {
        dimension: build_slice_report(per_record_rows, dimension)
        for dimension in [
            "question_id",
            "year_bucket",
            "population_signature_bucket",
            "option_count",
            "question_length_bucket",
            "fallback_happened",
            "strict_json_success",
        ]
    }
    recommendation, stop_combiner = recommend_route(
        heuristic_overall,
        deepseek_overall,
        slice_report,
    )
    summary_lines = [
        "# DeepSeek v3 Product Route Summary",
        "",
        f"- Recommendation: `{recommendation}`",
        f"- Stop combiner research: `{str(stop_combiner).lower()}`",
        f"- Overall heuristic JS: `{heuristic_overall['js_divergence']:.4f}`",
        f"- Overall DeepSeek JS: `{deepseek_overall['js_divergence']:.4f}`",
        f"- DeepSeek fallback rate: `{deepseek_ops['fallback_rate']:.4f}`",
        f"- DeepSeek JSON compliance rate: `{deepseek_ops['json_compliance_rate']:.4f}`",
        "",
        "## Product recommendation",
    ]
    if recommendation == "A":
        summary_lines.extend(
            [
                "- `deepseek_direct` should be the MVP main model.",
                "- `heuristic` should remain the fallback path.",
                "- Combiner research can be paused for now.",
                "",
                "## Productization checklist",
                "- add auth and per-key rate limiting",
                "- add persistent request logging and tracing",
                "- add monitoring for fallback rate, invalid rate, latency, and cost",
                "- add dashboarding for token usage and cost by tenant",
                "- add API docs and operational runbooks",
            ]
        )
    elif recommendation == "B":
        summary_lines.extend(
            [
                "- `heuristic` should remain the default.",
                "- `deepseek_direct` should stay as an optional enhancement path.",
                "- Combiner research should stay paused until a new benchmark direction exists.",
            ]
        )
    else:
        summary_lines.extend(
            [
                "- Use simple routing rather than a learned combiner.",
                "- `deepseek_direct` is useful but not uniformly dominant across slices.",
                "- Keep `heuristic` as a strong fallback and selective path.",
            ]
        )
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    report = {
        "formal_manifest": args.formal_manifest,
        "anti_leakage_status": anti_leakage_audit["status"],
        "env_file_used": env_summary["path"],
        "env_providers_available": env_summary["providers"],
        "generation_parameters": {
            "llm_provider": "openai_compatible",
            "llm_model": "deepseek-chat",
            "base_url": engine.deepseek_base_url,
            "strict_json": True,
            "final_text_only": True,
            "llm_num_samples": args.llm_num_samples,
            "llm_top_p": args.llm_top_p,
            "llm_max_tokens": args.llm_max_tokens,
            "request_timeout_seconds": args.request_timeout_seconds,
            "max_retry_attempts": args.max_retry_attempts,
        },
        "results": overall_results,
        "actual_strategy_breakdown": dict(
            Counter(row["deepseek"]["actual_strategy_used"] for row in per_record_rows)
        ),
        "fallback_reason_breakdown": dict(
            Counter(row["deepseek"]["fallback_reason"] or "none" for row in per_record_rows)
        ),
        "per_record_results": per_record_rows,
    }
    heuristic_result: dict[str, Any] = overall_results[0]
    deepseek_result: dict[str, Any] = overall_results[1]
    comparison = {
        "heuristic_prompt_only": heuristic_result,
        "llm_direct_option_probabilities": deepseek_result,
        "delta_js_deepseek_minus_heuristic": float(
            float(deepseek_result["js_divergence"]) - float(heuristic_result["js_divergence"])
        ),
        "delta_mae_deepseek_minus_heuristic": float(
            float(deepseek_result["probability_mae"])
            - float(heuristic_result["probability_mae"])
        ),
        "delta_rmse_deepseek_minus_heuristic": float(
            float(deepseek_result["probability_rmse"])
            - float(heuristic_result["probability_rmse"])
        ),
        "delta_top1_deepseek_minus_heuristic": float(
            float(deepseek_result["top_option_accuracy"])
            - float(heuristic_result["top_option_accuracy"])
        ),
        "recommendation": recommendation,
        "stop_combiner_research": stop_combiner,
    }

    write_json(args.output, report)
    write_json(args.comparison_output, comparison)
    write_json(args.slice_output, slice_report)
    print(f"Wrote DeepSeek v3 benchmark report to {args.output}")
    print(f"Wrote DeepSeek v3 comparison report to {args.comparison_output}")
    print(f"Wrote DeepSeek v3 slice report to {args.slice_output}")
    print(f"Wrote DeepSeek v3 summary to {args.summary_output}")


if __name__ == "__main__":
    main()
