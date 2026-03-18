from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from spbce.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    provider_summaries: list[dict[str, Any]] = []
    for input_path in args.inputs:
        report = read_json(input_path)
        results = report["results"]
        scored_results = [item for item in results if item.get("scored_record_count", 0) > 0]
        best_result = min(
            scored_results or results,
            key=lambda item: item["js_divergence"],
        )
        provider_summaries.append(
            {
                "report_path": str(Path(input_path)),
                "provider": report["llm_provider"],
                "model": report["llm_model"],
                "best_predictor": best_result["predictor"],
                "best_js_divergence": best_result["js_divergence"],
                "best_prompt_sensitivity_js": best_result["prompt_sensitivity_js"],
                "best_invalid_output_rate": best_result["invalid_output_rate"],
                "best_scored_record_rate": best_result.get("scored_record_rate", 0.0),
            }
        )
        for item in results:
            rows.append(
                {
                    "report_path": str(Path(input_path)),
                    "provider": report["llm_provider"],
                    "model": report["llm_model"],
                    "predictor": item["predictor"],
                    "js_divergence": item["js_divergence"],
                    "probability_mae": item["probability_mae"],
                    "probability_rmse": item["probability_rmse"],
                    "top_option_accuracy": item["top_option_accuracy"],
                    "sampling_variance": item["sampling_variance"],
                    "prompt_sensitivity_js": item["prompt_sensitivity_js"],
                    "final_text_present_rate": item["final_text_present_rate"],
                    "invalid_output_rate": item["invalid_output_rate"],
                    "json_compliance_rate": item.get("json_compliance_rate", 0.0),
                    "parser_failure_rate": item.get("parser_failure_rate", 0.0),
                    "estimated_api_cost_usd": item.get("estimated_api_cost_usd", 0.0),
                    "average_latency_ms_per_request": item.get(
                        "average_latency_ms_per_request", 0.0
                    ),
                    "scored_record_rate": item.get("scored_record_rate", 0.0),
                }
            )

    write_json(
        args.output,
        {
            "reports": [str(Path(path)) for path in args.inputs],
            "rows": rows,
            "provider_summaries": provider_summaries,
        },
    )
    print(f"Wrote model comparison report to {args.output}")


if __name__ == "__main__":
    main()
