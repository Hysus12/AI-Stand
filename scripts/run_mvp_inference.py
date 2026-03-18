from __future__ import annotations

import argparse
import json

from spbce.inference.mvp import MvpInferenceEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-text", required=True)
    parser.add_argument("--options", nargs="+", required=True)
    parser.add_argument("--population-text", required=True)
    parser.add_argument(
        "--strategy",
        choices=[
            "heuristic",
            "deepseek_direct",
            "weighted_hybrid_025",
            "learned_hybrid",
            "best_hybrid",
        ],
        default="deepseek_direct",
    )
    parser.add_argument("--env-file", default="api.env")
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-retry-attempts", type=int, default=1)
    parser.add_argument("--hybrid-strategy", default="weighted_average")
    parser.add_argument("--hybrid-llm-weight", type=float, default=0.25)
    parser.add_argument("--learned-combiner-artifact")
    args = parser.parse_args()

    engine = MvpInferenceEngine(
        env_file=args.env_file,
        request_timeout_seconds=args.request_timeout_seconds,
        max_retry_attempts=args.max_retry_attempts,
        weighted_hybrid_strategy=args.hybrid_strategy,
        weighted_hybrid_config={"llm_weight": args.hybrid_llm_weight},
        learned_combiner_artifact=args.learned_combiner_artifact,
    )
    request = engine.build_request(
        question_text=args.question_text,
        options=args.options,
        population_text=args.population_text,
    )
    response = engine.predict(request, strategy=args.strategy)
    print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
