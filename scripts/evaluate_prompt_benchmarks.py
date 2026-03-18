from __future__ import annotations

import argparse
import time
from typing import Any

from spbce.baselines.direct_probability_llm import DirectProbabilityLlmBaseline
from spbce.baselines.persona_llm import LocalLlmPersonaBaseline
from spbce.baselines.prompt_only import PromptOnlyPersonaBaseline
from spbce.data.datasets import load_survey_records
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
    build_few_shot_pool,
    derive_contaminated_question_blacklist,
    validate_formal_holdout,
)


def filter_records(records: list[SurveyRecord], record_ids: list[str]) -> list[SurveyRecord]:
    allowed = set(record_ids)
    return [record for record in records if record.record_id in allowed]


def evaluate_prompt_model(
    records: list[SurveyRecord],
    baseline_name: str,
    predictor: PromptOnlyPersonaBaseline | LocalLlmPersonaBaseline | DirectProbabilityLlmBaseline,
    few_shot: bool = False,
) -> dict[str, Any]:
    metrics: dict[str, list[float]] = {
        "js_divergence": [],
        "safe_kl_divergence": [],
        "probability_mae": [],
        "probability_rmse": [],
        "top_option_accuracy": [],
        "sampling_variance": [],
        "prompt_sensitivity_js": [],
        "final_text_present_rate": [],
        "used_thinking_fallback_rate": [],
        "invalid_output_rate": [],
        "json_compliance_rate": [],
        "parser_failure_rate": [],
    }
    diagnostics: dict[str, list[dict[str, Any]]] = {
        "raw_response_examples": [],
        "raw_response_diversity_by_record": [],
    }
    scored_record_ids: list[str] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_estimated_api_cost_usd = 0.0
    request_latencies_ms: list[float] = []
    started = time.perf_counter()
    for record in records:
        request = PredictSurveyRequest(
            question_text=record.question_text,
            options=record.options,
            population_text=record.population_text,
            population_struct=record.population_struct,
            context=SurveyContext(product_category=record.domain),
        )
        if isinstance(predictor, LocalLlmPersonaBaseline):
            result = predictor.sample_distribution(request, few_shot=few_shot)
            predicted = result["distribution"]
            metrics["sampling_variance"].append(float(result["sampling_variance"]))
            metrics["prompt_sensitivity_js"].append(
                float(result["prompt_paraphrase_sensitivity_js"])
            )
            metrics["final_text_present_rate"].append(float(result["final_text_present_rate"]))
            metrics["used_thinking_fallback_rate"].append(
                float(result["used_thinking_fallback_rate"])
            )
            metrics["invalid_output_rate"].append(float(result["invalid_output_rate"]))
            metrics["json_compliance_rate"].append(float(result["json_compliance_rate"]))
            metrics["parser_failure_rate"].append(float(result["parser_failure_rate"]))
            total_input_tokens += int(result["total_input_tokens"])
            total_output_tokens += int(result["total_output_tokens"])
            total_estimated_api_cost_usd += float(result["estimated_api_cost_usd"])
            request_latencies_ms.extend(
                [float(value) for value in result.get("request_latencies_ms", [])]
            )
            llm_examples = result["raw_response_examples"]
            llm_diversity = result["raw_response_diversity"]
            scorable = bool(result["scorable"]) and predicted is not None
        else:
            predicted = predictor.predict_proba(request)
            metrics["sampling_variance"].append(0.0)
            metrics["prompt_sensitivity_js"].append(0.0)
            metrics["final_text_present_rate"].append(1.0)
            metrics["used_thinking_fallback_rate"].append(0.0)
            metrics["invalid_output_rate"].append(0.0)
            metrics["json_compliance_rate"].append(0.0)
            metrics["parser_failure_rate"].append(0.0)
            llm_examples = []
            llm_diversity = None
            scorable = True
        if scorable and predicted is not None:
            observed = record.observed_distribution
            metrics["js_divergence"].append(js_divergence(predicted, observed))
            metrics["safe_kl_divergence"].append(safe_kl_divergence(predicted, observed))
            metrics["probability_mae"].append(probability_mae(predicted, observed))
            metrics["probability_rmse"].append(probability_rmse(predicted, observed))
            metrics["top_option_accuracy"].append(top_option_accuracy(predicted, observed))
            scored_record_ids.append(record.record_id)
        if llm_examples:
            diagnostics["raw_response_examples"].extend(
                [
                    {
                        "record_id": record.record_id,
                        "question_text": record.question_text,
                        **example,
                    }
                    for example in llm_examples
                ]
            )
        if llm_diversity is not None:
            diagnostics["raw_response_diversity_by_record"].append(
                {
                    "record_id": record.record_id,
                    "question_text": record.question_text,
                    **llm_diversity,
                }
            )

    report: dict[str, Any] = {"predictor": baseline_name}
    for key, values in metrics.items():
        report[key] = float(sum(values) / len(values)) if values else 0.0
    report["scored_record_count"] = len(scored_record_ids)
    report["scored_record_rate"] = float(len(scored_record_ids) / len(records)) if records else 0.0
    report["total_input_tokens"] = total_input_tokens
    report["total_output_tokens"] = total_output_tokens
    report["estimated_api_cost_usd"] = total_estimated_api_cost_usd
    report["estimated_cost_per_valid_record_usd"] = (
        float(total_estimated_api_cost_usd / len(scored_record_ids)) if scored_record_ids else 0.0
    )
    report["total_runtime_seconds"] = time.perf_counter() - started
    report["average_latency_ms_per_request"] = (
        float(sum(request_latencies_ms) / len(request_latencies_ms))
        if request_latencies_ms
        else 0.0
    )
    report["p50_latency_ms"] = (
        float(sorted(request_latencies_ms)[len(request_latencies_ms) // 2])
        if request_latencies_ms
        else 0.0
    )
    report["p95_latency_ms"] = (
        float(sorted(request_latencies_ms)[max(0, int(len(request_latencies_ms) * 0.95) - 1)])
        if request_latencies_ms
        else 0.0
    )
    if isinstance(predictor, LocalLlmPersonaBaseline):
        report["generation_config"] = predictor.generation_config()
        report["raw_response_examples"] = diagnostics["raw_response_examples"][:20]
        report["raw_response_diversity_by_record"] = diagnostics["raw_response_diversity_by_record"]
        if predictor.pool_summary:
            report["few_shot_pool"] = predictor.pool_summary
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", default="reports/benchmarks/prompt_benchmark.json")
    parser.add_argument("--max-records", type=int, default=12)
    parser.add_argument("--llm-model", default="google/flan-t5-small")
    parser.add_argument(
        "--llm-provider",
        choices=["auto", "local", "anthropic_compatible", "openai_compatible", "openai"],
        default="auto",
    )
    parser.add_argument("--env-file")
    parser.add_argument("--contamination-audit-report")
    parser.add_argument("--formal-manifest")
    parser.add_argument("--anti-leakage-audit-output")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-num-samples", type=int, default=24)
    parser.add_argument("--llm-top-p", type=float, default=0.95)
    parser.add_argument("--llm-max-tokens", type=int, default=16)
    parser.add_argument("--llm-thinking", action="store_true")
    parser.add_argument("--llm-reasoning-effort", default="none")
    parser.add_argument("--strict-direct-json", action="store_true")
    args = parser.parse_args()

    initialize_runtime_env(args.env_file)
    env_summary = get_provider_environment_summary(args.env_file)
    records = load_survey_records(args.input)
    split_manifest = read_json(args.splits)
    anti_leakage_audit: dict[str, Any] | None = None
    few_shot_pool_summary: dict[str, Any] = {}
    if args.formal_manifest:
        formal_manifest = read_json(args.formal_manifest)
        if not args.contamination_audit_report:
            raise RuntimeError("--contamination-audit-report is required with --formal-manifest")
        blacklist = derive_contaminated_question_blacklist(args.contamination_audit_report)
        anti_leakage_audit = validate_formal_holdout(records, formal_manifest, blacklist)
        if args.anti_leakage_audit_output:
            write_json(args.anti_leakage_audit_output, anti_leakage_audit)
        if anti_leakage_audit["status"] != "pass":
            raise RuntimeError("formal holdout anti-leakage validation failed")
        train_records = filter_records(records, formal_manifest["train_record_ids"])
        test_records = filter_records(records, formal_manifest["test_record_ids"])
        few_shot_pool_records, formal_pool_summary = build_few_shot_pool(
            train_records, test_records
        )
        few_shot_pool_summary = formal_pool_summary.to_dict()
    else:
        train_records = filter_records(records, split_manifest["resolved_record_ids"]["train"])
        test_records = filter_records(records, split_manifest["resolved_record_ids"]["test"])[
            : args.max_records
        ]
        few_shot_pool_records = train_records

    heuristic = PromptOnlyPersonaBaseline(backend="heuristic")
    llm_zero_shot = LocalLlmPersonaBaseline(
        model_name=args.llm_model,
        provider=args.llm_provider,
        env_file=args.env_file,
        anthropic_base_url=args.llm_base_url,
        openai_base_url=args.llm_base_url,
        num_samples=args.llm_num_samples,
        top_p=args.llm_top_p,
        max_new_tokens=args.llm_max_tokens,
        thinking_enabled=args.llm_thinking,
        reasoning_effort=args.llm_reasoning_effort,
    ).fit(train_records)
    llm_few_shot = LocalLlmPersonaBaseline(
        model_name=args.llm_model,
        provider=args.llm_provider,
        env_file=args.env_file,
        anthropic_base_url=args.llm_base_url,
        openai_base_url=args.llm_base_url,
        num_samples=args.llm_num_samples,
        top_p=args.llm_top_p,
        max_new_tokens=args.llm_max_tokens,
        thinking_enabled=args.llm_thinking,
        reasoning_effort=args.llm_reasoning_effort,
    ).fit(few_shot_pool_records)
    llm_few_shot.pool_summary = few_shot_pool_summary
    direct_probability_baseline = DirectProbabilityLlmBaseline(
        model_name=args.llm_model,
        provider=args.llm_provider,
        env_file=args.env_file,
        anthropic_base_url=args.llm_base_url,
        openai_base_url=args.llm_base_url,
        num_samples=args.llm_num_samples,
        top_p=args.llm_top_p,
        max_new_tokens=args.llm_max_tokens,
        thinking_enabled=args.llm_thinking,
        reasoning_effort=args.llm_reasoning_effort,
        strict_json_only=args.strict_direct_json or bool(args.formal_manifest),
    ).fit(train_records)

    results = [
        evaluate_prompt_model(test_records, "heuristic_prompt_only", heuristic),
        evaluate_prompt_model(test_records, "llm_zero_shot_persona", llm_zero_shot, few_shot=False),
        evaluate_prompt_model(test_records, "llm_few_shot_persona", llm_few_shot, few_shot=True),
        evaluate_prompt_model(
            test_records,
            "llm_direct_option_probabilities",
            direct_probability_baseline,
            few_shot=False,
        ),
    ]
    write_json(
        args.output,
        {
            "split": split_manifest["strategy"],
            "num_test_records": len(test_records),
            "env_file_used": env_summary["path"],
            "env_providers_available": env_summary["providers"],
            "llm_provider": llm_zero_shot._provider_name(),
            "llm_model": llm_zero_shot.model_name,
            "base_url": llm_zero_shot._resolved_base_url(),
            "formal_manifest": args.formal_manifest,
            "anti_leakage_audit_output": args.anti_leakage_audit_output,
            "anti_leakage_audit_status": anti_leakage_audit["status"]
            if anti_leakage_audit
            else None,
            "generation_parameters": {
                "top_p": args.llm_top_p,
                "max_tokens": args.llm_max_tokens,
                "thinking_sent": args.llm_thinking,
                "reasoning_effort": args.llm_reasoning_effort,
                "strict_direct_json": args.strict_direct_json or bool(args.formal_manifest),
                "persona_num_samples_per_prompt": args.llm_num_samples,
                "persona_temperature_base": 0.9,
                "persona_temperature_step": 0.05,
                "direct_probability_temperature_base": 0.2,
                "direct_probability_temperature_step": 0.05,
            },
            "results": results,
        },
    )
    print(f"Wrote prompt benchmark report to {args.output}")


if __name__ == "__main__":
    main()
