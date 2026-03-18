from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from spbce.baselines.direct_probability_llm import DirectProbabilityLlmBaseline
from spbce.baselines.learned_combiner import LearnedHybridCombiner, optimal_mixing_weight
from spbce.baselines.prompt_only import PromptOnlyPersonaBaseline
from spbce.data.datasets import load_survey_records
from spbce.metrics.distributions import (
    js_divergence,
    probability_mae,
    probability_rmse,
    top_option_accuracy,
)
from spbce.schema.api import PredictSurveyRequest, SurveyContext
from spbce.schema.canonical import SurveyRecord
from spbce.utils.io import read_json, write_json


def filter_records(records: list[SurveyRecord], record_ids: list[str]) -> list[SurveyRecord]:
    allowed = set(record_ids)
    return [record for record in records if record.record_id in allowed]


def evaluate_rows(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    predicted_rows = [row[key] for row in rows]
    observed_rows = [row["observed_distribution"] for row in rows]
    return {
        "js_divergence": float(
            sum(
                js_divergence(predicted, observed)
                for predicted, observed in zip(predicted_rows, observed_rows, strict=True)
            )
            / len(rows)
        ),
        "probability_mae": float(
            sum(
                probability_mae(predicted, observed)
                for predicted, observed in zip(predicted_rows, observed_rows, strict=True)
            )
            / len(rows)
        ),
        "probability_rmse": float(
            sum(
                probability_rmse(predicted, observed)
                for predicted, observed in zip(predicted_rows, observed_rows, strict=True)
            )
            / len(rows)
        ),
        "top_option_accuracy": float(
            sum(
                top_option_accuracy(predicted, observed)
                for predicted, observed in zip(predicted_rows, observed_rows, strict=True)
            )
            / len(rows)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--env-file", default="api.env")
    parser.add_argument("--llm-base-url", default="https://api.deepseek.com")
    parser.add_argument("--output-artifact", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--llm-num-samples", type=int, default=4)
    parser.add_argument("--llm-max-tokens", type=int, default=128)
    args = parser.parse_args()

    records = load_survey_records(args.input)
    manifest = read_json(args.split_manifest)
    train_records = filter_records(records, manifest["train_record_ids"])
    dev_records = filter_records(records, manifest["dev_record_ids"])

    heuristic = PromptOnlyPersonaBaseline(backend="heuristic")
    direct = DirectProbabilityLlmBaseline(
        model_name="deepseek-chat",
        provider="openai_compatible",
        env_file=args.env_file,
        openai_base_url=args.llm_base_url,
        num_samples=args.llm_num_samples,
        max_new_tokens=args.llm_max_tokens,
        strict_json_only=True,
        reasoning_effort="none",
    )
    combiner = LearnedHybridCombiner(
        heuristic_predictor=heuristic,
        llm_predictor=direct,
    )

    def build_rows(source_records: list[SurveyRecord]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for record in source_records:
            request = PredictSurveyRequest(
                question_text=record.question_text,
                options=record.options,
                population_text=record.population_text,
                population_struct=record.population_struct,
                context=SurveyContext(product_category=record.domain),
            )
            heuristic_distribution = heuristic.predict_proba(request)
            llm_result = direct.sample_distribution(request, few_shot=False)
            llm_distribution = llm_result["distribution"]
            if llm_distribution is None:
                continue
            feature_row, feature_dict = combiner._build_feature_row(
                request=request,
                heuristic_distribution=heuristic_distribution,
                llm_result=llm_result,
            )
            rows.append(
                {
                    "record_id": record.record_id,
                    "question_id": record.question_id,
                    "wave_id": record.wave_id,
                    "features": feature_dict,
                    "feature_row": feature_row,
                    "observed_distribution": record.observed_distribution,
                    "heuristic_distribution": heuristic_distribution,
                    "llm_distribution": llm_distribution,
                    "weighted_025_distribution": [
                        (0.75 * heuristic_probability) + (0.25 * llm_probability)
                        for heuristic_probability, llm_probability in zip(
                            heuristic_distribution, llm_distribution, strict=True
                        )
                    ],
                    "json_compliance_rate": llm_result["json_compliance_rate"],
                    "invalid_output_rate": llm_result["invalid_output_rate"],
                    "estimated_api_cost_usd": llm_result["estimated_api_cost_usd"],
                    "average_latency_ms_per_request": (
                        float(
                            sum(llm_result["request_latencies_ms"])
                            / len(llm_result["request_latencies_ms"])
                        )
                        if llm_result["request_latencies_ms"]
                        else 0.0
                    ),
                    "target_weight": optimal_mixing_weight(
                        heuristic_distribution=heuristic_distribution,
                        llm_distribution=llm_distribution,
                        observed_distribution=record.observed_distribution,
                    ),
                }
            )
        return rows

    train_rows = build_rows(train_records)
    dev_rows = build_rows(dev_records)
    combiner.fit_rows(train_rows)
    combiner.training_metadata = {
        "split_manifest": args.split_manifest,
        "train_row_count": len(train_rows),
        "dev_row_count": len(dev_rows),
    }
    Path(args.output_artifact).parent.mkdir(parents=True, exist_ok=True)
    combiner.save(args.output_artifact)

    for row in dev_rows:
        dev_record = next(record for record in dev_records if record.record_id == row["record_id"])
        request = PredictSurveyRequest(
            question_text=dev_record.question_text,
            options=dev_record.options,
            population_text=dev_record.population_text,
            population_struct=dev_record.population_struct,
            context=SurveyContext(),
        )
        weight, _ = combiner.predict_weight_from_result(
            request=request,
            heuristic_distribution=row["heuristic_distribution"],
            llm_result={
                "distribution": row["llm_distribution"],
                "json_compliance_rate": row["json_compliance_rate"],
                "invalid_output_rate": row["invalid_output_rate"],
                "final_text_present_rate": 1.0,
            },
        )
        row["predicted_weight"] = weight
        row["learned_distribution"] = [
            ((1.0 - weight) * heuristic_probability) + (weight * llm_probability)
            for heuristic_probability, llm_probability in zip(
                row["heuristic_distribution"], row["llm_distribution"], strict=True
            )
        ]

    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dev_rows:
        by_question[row["question_id"]].append(row)

    report = {
        "artifact_path": args.output_artifact,
        "split_manifest": args.split_manifest,
        "train_row_count": len(train_rows),
        "dev_row_count": len(dev_rows),
        "feature_names": combiner.feature_names,
        "baselines": {
            "heuristic_prompt_only": evaluate_rows(dev_rows, "heuristic_distribution"),
            "deepseek_direct": evaluate_rows(dev_rows, "llm_distribution"),
            "weighted_llm_0.25": evaluate_rows(dev_rows, "weighted_025_distribution"),
            "learned_combiner_v1": evaluate_rows(dev_rows, "learned_distribution"),
        },
        "average_predicted_weight_on_dev": float(
            sum(row["predicted_weight"] for row in dev_rows) / len(dev_rows)
        )
        if dev_rows
        else 0.0,
        "dev_cost_summary": {
            "estimated_api_cost_usd": float(sum(row["estimated_api_cost_usd"] for row in dev_rows)),
            "average_latency_ms_per_request": float(
                sum(row["average_latency_ms_per_request"] for row in dev_rows) / len(dev_rows)
            )
            if dev_rows
            else 0.0,
        },
        "dev_results_by_question_id": {
            question_id: {
                "row_count": len(question_rows),
                "heuristic_prompt_only": evaluate_rows(question_rows, "heuristic_distribution"),
                "deepseek_direct": evaluate_rows(question_rows, "llm_distribution"),
                "weighted_llm_0.25": evaluate_rows(question_rows, "weighted_025_distribution"),
                "learned_combiner_v1": evaluate_rows(question_rows, "learned_distribution"),
            }
            for question_id, question_rows in sorted(by_question.items())
        },
    }
    write_json(args.output_report, report)
    print(f"Wrote learned combiner artifact to {args.output_artifact}")
    print(f"Wrote learned combiner report to {args.output_report}")


if __name__ == "__main__":
    main()
