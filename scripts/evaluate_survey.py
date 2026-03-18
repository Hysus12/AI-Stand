from __future__ import annotations

import argparse

import joblib

from spbce.baselines.majority import MajorityDistributionBaseline
from spbce.baselines.prompt_only import PromptOnlyPersonaBaseline
from spbce.baselines.subgroup import SubgroupMarginalBaseline
from spbce.baselines.topic import TopicOnlyBaseline
from spbce.data.datasets import load_survey_records
from spbce.metrics.distributions import (
    js_divergence,
    probability_mae,
    probability_rmse,
    safe_kl_divergence,
    top_option_accuracy,
)
from spbce.schema.api import PredictSurveyRequest
from spbce.schema.canonical import SurveyRecord
from spbce.utils.io import read_json, write_json


def filter_records(records: list[SurveyRecord], record_ids: list[str]) -> list[SurveyRecord]:
    allowed = set(record_ids)
    return [record for record in records if record.record_id in allowed]


def evaluate_predictions(
    records: list[SurveyRecord], predictor_name: str, predictor
) -> dict[str, float]:
    metrics = {
        "js_divergence": [],
        "safe_kl_divergence": [],
        "probability_mae": [],
        "probability_rmse": [],
        "top_option_accuracy": [],
    }
    for record in records:
        request = PredictSurveyRequest(
            question_text=record.question_text,
            options=record.options,
            population_text=record.population_text,
            population_struct=record.population_struct,
            context={"product_category": record.domain},
        )
        predicted = predictor.predict_proba(request)
        observed = record.observed_distribution
        metrics["js_divergence"].append(js_divergence(predicted, observed))
        metrics["safe_kl_divergence"].append(safe_kl_divergence(predicted, observed))
        metrics["probability_mae"].append(probability_mae(predicted, observed))
        metrics["probability_rmse"].append(probability_rmse(predicted, observed))
        metrics["top_option_accuracy"].append(top_option_accuracy(predicted, observed))
    return {"predictor": predictor_name} | {
        metric_name: float(sum(values) / len(values)) for metric_name, values in metrics.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", default="reports/benchmarks/survey_eval.json")
    parser.add_argument("--artifact-dir", default="data/processed/artifacts")
    args = parser.parse_args()

    records = load_survey_records(args.input)
    split_manifest = read_json(args.splits)
    test_records = filter_records(records, split_manifest["resolved_record_ids"]["test"])

    majority: MajorityDistributionBaseline = joblib.load(
        f"{args.artifact_dir}/majority_baseline.joblib"
    )
    subgroup: SubgroupMarginalBaseline = joblib.load(
        f"{args.artifact_dir}/subgroup_baseline.joblib"
    )
    topic: TopicOnlyBaseline = joblib.load(f"{args.artifact_dir}/topic_baseline.joblib")
    prompt: PromptOnlyPersonaBaseline = joblib.load(
        f"{args.artifact_dir}/demo_prompt_baseline.joblib"
    )
    survey_artifact = joblib.load(f"{args.artifact_dir}/demo_survey_model.joblib")
    survey_model = (
        survey_artifact["survey_model"] if isinstance(survey_artifact, dict) else survey_artifact
    )
    scaler = (
        survey_artifact.get("temperature_scaler") if isinstance(survey_artifact, dict) else None
    )

    rows = [
        evaluate_predictions(test_records, "majority_distribution", majority),
        evaluate_predictions(test_records, "subgroup_marginal", subgroup),
        evaluate_predictions(test_records, "topic_only", topic),
        evaluate_predictions(test_records, "prompt_only", prompt),
    ]
    raw_row = evaluate_predictions(test_records, "simple_supervised", survey_model)
    if scaler is not None:

        class CalibratedPredictor:
            def __init__(self, model, scaler) -> None:
                self.model = model
                self.scaler = scaler

            def predict_proba(self, request: PredictSurveyRequest) -> list[float]:
                return self.scaler.apply(self.model.predict_proba(request))

        rows.append(raw_row)
        rows.append(
            evaluate_predictions(
                test_records,
                "simple_supervised_calibrated",
                CalibratedPredictor(survey_model, scaler),
            )
        )
    else:
        rows.append(raw_row)

    write_json(args.output, {"split": split_manifest["strategy"], "results": rows})
    print(f"Wrote survey evaluation report to {args.output}")


if __name__ == "__main__":
    main()
