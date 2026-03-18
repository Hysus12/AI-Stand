from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spbce.baselines.direct_probability_llm import DirectProbabilityLlmBaseline
from spbce.baselines.hybrid import HybridPredictor
from spbce.baselines.learned_combiner import LearnedHybridCombiner
from spbce.baselines.prompt_only import PromptOnlyPersonaBaseline
from spbce.schema.api import PredictSurveyRequest, SurveyContext
from spbce.schema.canonical import PopulationStruct


def top_option_from_distribution(options: list[str], distribution: list[float]) -> str:
    if not options:
        return ""
    best_index = max(range(len(options)), key=lambda index: distribution[index])
    return options[best_index]


@dataclass(slots=True)
class MvpInferenceEngine:
    env_file: str | None = None
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"
    llm_num_samples: int = 4
    llm_top_p: float = 0.95
    llm_max_tokens: int = 128
    default_strategy: str = "heuristic"
    weighted_hybrid_strategy: str = "weighted_average"
    weighted_hybrid_config: dict[str, Any] | None = None
    learned_combiner_artifact: str | None = None
    heuristic: PromptOnlyPersonaBaseline = field(init=False, repr=False)
    deepseek_direct: DirectProbabilityLlmBaseline = field(init=False, repr=False)
    weighted_hybrid_025: HybridPredictor = field(init=False, repr=False)
    learned_hybrid: LearnedHybridCombiner | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.heuristic = PromptOnlyPersonaBaseline(backend="heuristic")
        self.deepseek_direct = DirectProbabilityLlmBaseline(
            model_name=self.deepseek_model,
            provider="openai_compatible",
            env_file=self.env_file,
            openai_base_url=self.deepseek_base_url,
            num_samples=self.llm_num_samples,
            top_p=self.llm_top_p,
            max_new_tokens=self.llm_max_tokens,
            reasoning_effort="none",
            strict_json_only=True,
        )
        hybrid_config = self.weighted_hybrid_config or {"llm_weight": 0.25}
        self.weighted_hybrid_025 = HybridPredictor(
            name="weighted_hybrid_025",
            heuristic_predictor=self.heuristic,
            llm_predictor=self.deepseek_direct,
            strategy=self.weighted_hybrid_strategy,
            config=hybrid_config,
        )
        if self.learned_combiner_artifact and Path(self.learned_combiner_artifact).exists():
            self.learned_hybrid = LearnedHybridCombiner.load(
                path=self.learned_combiner_artifact,
                heuristic_predictor=self.heuristic,
                llm_predictor=self.deepseek_direct,
            )

    def build_request(
        self,
        question_text: str,
        options: list[str],
        population_text: str,
        population_struct: dict[str, Any] | None = None,
    ) -> PredictSurveyRequest:
        return PredictSurveyRequest(
            question_text=question_text,
            options=options,
            population_text=population_text,
            population_struct=PopulationStruct.model_validate(population_struct or {}),
            context=SurveyContext(),
        )

    def predict(
        self, request: PredictSurveyRequest, strategy: str | None = None
    ) -> dict[str, Any]:
        predictor: DirectProbabilityLlmBaseline | HybridPredictor | LearnedHybridCombiner
        resolved_strategy = strategy or self.default_strategy
        deprecated_alias = False
        if resolved_strategy == "best_hybrid":
            resolved_strategy = "weighted_hybrid_025"
            deprecated_alias = True
        if resolved_strategy == "heuristic":
            distribution = self.heuristic.predict_proba(request)
            return {
                "distribution": {
                    option: float(probability)
                    for option, probability in zip(request.options, distribution, strict=True)
                },
                "top_option": top_option_from_distribution(request.options, distribution),
                "confidence": float(max(distribution)),
                "uncertainty": float(1.0 - max(distribution)),
                "invalid_flag": False,
                "source": "heuristic",
                "metadata": {
                    "strategy_used": "heuristic",
                    "model_name": None,
                    "generation_config": None,
                    "fallback_happened": False,
                    "output_type": "heuristic",
                    "cost_latency_summary": {
                        "estimated_api_cost_usd": 0.0,
                        "average_latency_ms_per_request": 0.0,
                        "total_input_tokens": 0,
                        "total_output_tokens": 0,
                    },
                },
            }

        if resolved_strategy == "deepseek_direct":
            predictor = self.deepseek_direct
            output_type = "llm"
        elif resolved_strategy == "weighted_hybrid_025":
            predictor = self.weighted_hybrid_025
            output_type = "hybrid"
        elif resolved_strategy == "learned_hybrid":
            if self.learned_hybrid is None:
                raise RuntimeError("learned_hybrid requested but no combiner artifact was loaded")
            predictor = self.learned_hybrid
            output_type = "learned_hybrid"
        else:
            raise ValueError(f"Unsupported strategy: {resolved_strategy}")
        result = predictor.sample_distribution(request, few_shot=False)
        distribution = result["distribution"]
        invalid_flag = not bool(result["scorable"]) or distribution is None
        distribution_values = distribution or [1.0 / len(request.options)] * len(request.options)
        return {
            "distribution": {
                option: float(probability)
                for option, probability in zip(
                    request.options, distribution_values, strict=True
                )
            },
            "top_option": top_option_from_distribution(request.options, distribution_values),
            "confidence": float(max(distribution_values)),
            "uncertainty": float(1.0 - max(distribution_values)),
            "invalid_flag": invalid_flag,
            "source": resolved_strategy,
            "metadata": {
                "strategy_used": resolved_strategy,
                "strategy_alias_deprecated": deprecated_alias,
                "model_name": self.deepseek_model,
                "generation_config": predictor.generation_config(),
                "fallback_happened": bool(result.get("hybrid_fallback_used")),
                "output_type": output_type,
                "cost_latency_summary": {
                    "estimated_api_cost_usd": float(result.get("estimated_api_cost_usd", 0.0)),
                    "average_latency_ms_per_request": (
                        float(sum(result.get("request_latencies_ms", [])))
                        / len(result.get("request_latencies_ms", []))
                        if result.get("request_latencies_ms")
                        else 0.0
                    ),
                    "total_input_tokens": int(result.get("total_input_tokens", 0)),
                    "total_output_tokens": int(result.get("total_output_tokens", 0)),
                },
                "json_compliance_rate": float(result.get("json_compliance_rate", 0.0)),
                "invalid_output_rate": float(result.get("invalid_output_rate", 0.0)),
                "predicted_llm_weight": result.get("predicted_llm_weight"),
                "hybrid_decision": result.get("hybrid_decision"),
            },
        }
