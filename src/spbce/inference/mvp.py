from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spbce.baselines.direct_probability_llm import DirectProbabilityLlmBaseline
from spbce.baselines.hybrid import HybridPredictor, distribution_entropy
from spbce.baselines.learned_combiner import LearnedHybridCombiner
from spbce.baselines.prompt_only import PromptOnlyPersonaBaseline
from spbce.schema.api import (
    MvpCostLatencySummary,
    MvpPredictSurveyResponse,
    PredictSurveyRequest,
    SurveyContext,
)
from spbce.schema.canonical import PopulationStruct


def top_option_from_distribution(options: list[str], distribution: list[float]) -> str:
    if not options:
        return ""
    best_index = max(range(len(options)), key=lambda index: distribution[index])
    return options[best_index]


def distribution_margin(distribution: list[float]) -> float:
    if len(distribution) < 2:
        return float(distribution[0]) if distribution else 0.0
    ordered = sorted(distribution, reverse=True)
    return float(ordered[0] - ordered[1])


@dataclass(slots=True)
class MvpInferenceEngine:
    env_file: str | None = None
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"
    llm_num_samples: int = 4
    llm_top_p: float = 0.95
    llm_max_tokens: int = 128
    request_timeout_seconds: float = 30.0
    max_retry_attempts: int = 1
    default_strategy: str = "deepseek_direct"
    fallback_strategy: str = "heuristic"
    fallback_min_confidence: float | None = None
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
            request_timeout_seconds=self.request_timeout_seconds,
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

    def _cost_summary_from_result(
        self,
        result: dict[str, Any] | None = None,
        retry_count: int = 0,
        request_count_override: int | None = None,
    ) -> MvpCostLatencySummary:
        latencies = [float(value) for value in (result or {}).get("request_latencies_ms", [])]
        request_count = (
            request_count_override if request_count_override is not None else len(latencies)
        )
        total_latency_ms = float(sum(latencies))
        average_latency_ms = float(total_latency_ms / len(latencies)) if latencies else 0.0
        return MvpCostLatencySummary(
            total_input_tokens=int((result or {}).get("total_input_tokens", 0)),
            total_output_tokens=int((result or {}).get("total_output_tokens", 0)),
            estimated_api_cost_usd=float((result or {}).get("estimated_api_cost_usd", 0.0)),
            total_latency_ms=total_latency_ms,
            average_latency_ms_per_request=average_latency_ms,
            request_count=request_count,
            retry_count=retry_count,
        )

    def _confidence_summary(self, distribution: list[float]) -> dict[str, float]:
        return {
            "top1_probability": float(max(distribution)),
            "uncertainty": float(1.0 - max(distribution)),
            "entropy_normalized": float(distribution_entropy(distribution)),
            "top1_margin": float(distribution_margin(distribution)),
        }

    def _heuristic_response(
        self,
        request: PredictSurveyRequest,
        requested_strategy: str,
        fallback_reason: str | None = None,
        deprecated_alias: bool = False,
        attempted_result: dict[str, Any] | None = None,
        retry_count: int = 0,
    ) -> MvpPredictSurveyResponse:
        distribution = self.heuristic.predict_proba(request)
        confidence_summary = self._confidence_summary(distribution)
        return MvpPredictSurveyResponse(
            distribution={
                option: float(probability)
                for option, probability in zip(request.options, distribution, strict=True)
            },
            top_option=top_option_from_distribution(request.options, distribution),
            confidence=confidence_summary["top1_probability"],
            uncertainty=confidence_summary["uncertainty"],
            invalid_flag=False,
            requested_strategy=requested_strategy,
            actual_strategy_used="heuristic",
            fallback_happened=bool(fallback_reason),
            fallback_reason=fallback_reason,
            metadata={
                "strategy_alias_deprecated": deprecated_alias,
                "model_name": None,
                "actual_model_name": None,
                "requested_model_name": (
                    self.deepseek_model if requested_strategy != "heuristic" else None
                ),
                "generation_config": None,
                "output_type": "heuristic",
                "confidence_summary": confidence_summary,
                "json_compliance_rate": float(
                    (attempted_result or {}).get("json_compliance_rate", 0.0)
                ),
                "invalid_output_rate": float(
                    (attempted_result or {}).get("invalid_output_rate", 0.0)
                ),
                "parser_failure_rate": float(
                    (attempted_result or {}).get("parser_failure_rate", 0.0)
                ),
                "final_text_present_rate": float(
                    (attempted_result or {}).get("final_text_present_rate", 0.0)
                ),
                "attempt_summary": {
                    "requested_strategy": requested_strategy,
                    "actual_strategy_used": "heuristic",
                    "fallback_happened": bool(fallback_reason),
                    "fallback_reason": fallback_reason,
                    "retry_count": retry_count,
                },
            },
            cost_latency_summary=self._cost_summary_from_result(
                attempted_result,
                retry_count=retry_count,
                request_count_override=(
                    retry_count + 1
                    if attempted_result is None and requested_strategy != "heuristic"
                    else None
                ),
            ),
        )

    def _experimental_response(
        self,
        request: PredictSurveyRequest,
        requested_strategy: str,
        predictor: DirectProbabilityLlmBaseline | HybridPredictor | LearnedHybridCombiner,
        output_type: str,
        deprecated_alias: bool = False,
    ) -> MvpPredictSurveyResponse:
        result = predictor.sample_distribution(request, few_shot=False)
        distribution = result["distribution"]
        invalid_flag = not bool(result["scorable"]) or distribution is None
        distribution_values = distribution or [1.0 / len(request.options)] * len(request.options)
        confidence_summary = self._confidence_summary(distribution_values)
        return MvpPredictSurveyResponse(
            distribution={
                option: float(probability)
                for option, probability in zip(
                    request.options, distribution_values, strict=True
                )
            },
            top_option=top_option_from_distribution(request.options, distribution_values),
            confidence=confidence_summary["top1_probability"],
            uncertainty=confidence_summary["uncertainty"],
            invalid_flag=invalid_flag,
            requested_strategy=requested_strategy,
            actual_strategy_used=requested_strategy,
            fallback_happened=bool(result.get("hybrid_fallback_used")),
            fallback_reason=(
                "heuristic_fallback_inside_experimental_strategy"
                if bool(result.get("hybrid_fallback_used"))
                else None
            ),
            metadata={
                "strategy_alias_deprecated": deprecated_alias,
                "model_name": self.deepseek_model,
                "actual_model_name": self.deepseek_model,
                "requested_model_name": self.deepseek_model,
                "generation_config": predictor.generation_config(),
                "output_type": output_type,
                "confidence_summary": confidence_summary,
                "json_compliance_rate": float(result.get("json_compliance_rate", 0.0)),
                "invalid_output_rate": float(result.get("invalid_output_rate", 0.0)),
                "parser_failure_rate": float(result.get("parser_failure_rate", 0.0)),
                "final_text_present_rate": float(result.get("final_text_present_rate", 0.0)),
                "predicted_llm_weight": result.get("predicted_llm_weight"),
                "hybrid_decision": result.get("hybrid_decision"),
                "experimental": True,
                "attempt_summary": {
                    "requested_strategy": requested_strategy,
                    "actual_strategy_used": requested_strategy,
                    "fallback_happened": bool(result.get("hybrid_fallback_used")),
                    "fallback_reason": (
                        "heuristic_fallback_inside_experimental_strategy"
                        if bool(result.get("hybrid_fallback_used"))
                        else None
                    ),
                    "retry_count": 0,
                },
            },
            cost_latency_summary=self._cost_summary_from_result(result),
        )

    def _fallback_reason_for_direct_result(self, result: dict[str, Any]) -> str | None:
        distribution = result.get("distribution")
        if distribution is None:
            return "empty_output"
        if float(result.get("final_text_present_rate", 0.0)) < 1.0:
            return "empty_output"
        if float(result.get("json_compliance_rate", 0.0)) < 1.0:
            return "json_schema_validation_failed"
        if float(result.get("invalid_output_rate", 0.0)) > 0.0:
            return "invalid_output"
        if float(result.get("parser_failure_rate", 0.0)) > 0.0:
            return "parser_failure"
        if not bool(result.get("scorable")):
            return "invalid_output"
        if self.fallback_min_confidence is not None and float(max(distribution)) < float(
            self.fallback_min_confidence
        ):
            return "low_confidence"
        return None

    def _classify_exception(self, exc: Exception) -> str:
        message = str(exc).lower()
        if "timed out" in message or "timeout" in message:
            return "timeout"
        return "provider_error"

    def _predict_deepseek_direct(
        self,
        request: PredictSurveyRequest,
        requested_strategy: str,
        deprecated_alias: bool = False,
    ) -> MvpPredictSurveyResponse:
        last_result: dict[str, Any] | None = None
        fallback_reason: str | None = None
        last_exception_reason: str | None = None
        retry_count = 0

        for attempt_index in range(self.max_retry_attempts + 1):
            retry_count = attempt_index
            try:
                result = self.deepseek_direct.sample_distribution(request, few_shot=False)
            except Exception as exc:
                last_exception_reason = self._classify_exception(exc)
                fallback_reason = last_exception_reason
                if attempt_index < self.max_retry_attempts:
                    continue
                return self._heuristic_response(
                    request=request,
                    requested_strategy=requested_strategy,
                    fallback_reason=fallback_reason,
                    deprecated_alias=deprecated_alias,
                    attempted_result=last_result,
                    retry_count=retry_count,
                )

            last_result = result
            fallback_reason = self._fallback_reason_for_direct_result(result)
            if fallback_reason is None:
                distribution = result["distribution"]
                if distribution is None:
                    break
                confidence_summary = self._confidence_summary(distribution)
                return MvpPredictSurveyResponse(
                    distribution={
                        option: float(probability)
                        for option, probability in zip(
                            request.options, distribution, strict=True
                        )
                    },
                    top_option=top_option_from_distribution(request.options, distribution),
                    confidence=confidence_summary["top1_probability"],
                    uncertainty=confidence_summary["uncertainty"],
                    invalid_flag=False,
                    requested_strategy=requested_strategy,
                    actual_strategy_used="deepseek_direct",
                    fallback_happened=False,
                    fallback_reason=None,
                    metadata={
                        "strategy_alias_deprecated": deprecated_alias,
                        "model_name": self.deepseek_model,
                        "actual_model_name": self.deepseek_model,
                        "requested_model_name": self.deepseek_model,
                        "generation_config": self.deepseek_direct.generation_config(),
                        "output_type": "llm",
                        "confidence_summary": confidence_summary,
                        "json_compliance_rate": float(
                            result.get("json_compliance_rate", 0.0)
                        ),
                        "invalid_output_rate": float(
                            result.get("invalid_output_rate", 0.0)
                        ),
                        "parser_failure_rate": float(
                            result.get("parser_failure_rate", 0.0)
                        ),
                        "final_text_present_rate": float(
                            result.get("final_text_present_rate", 0.0)
                        ),
                        "attempt_summary": {
                            "requested_strategy": requested_strategy,
                            "actual_strategy_used": "deepseek_direct",
                            "fallback_happened": False,
                            "fallback_reason": None,
                            "retry_count": retry_count,
                        },
                    },
                    cost_latency_summary=self._cost_summary_from_result(
                        result,
                        retry_count=retry_count,
                    ),
                )
            if attempt_index < self.max_retry_attempts:
                continue

        return self._heuristic_response(
            request=request,
            requested_strategy=requested_strategy,
            fallback_reason=fallback_reason or last_exception_reason or "provider_error",
            deprecated_alias=deprecated_alias,
            attempted_result=last_result,
            retry_count=retry_count,
        )

    def predict(
        self, request: PredictSurveyRequest, strategy: str | None = None
    ) -> MvpPredictSurveyResponse:
        resolved_strategy = strategy or self.default_strategy
        deprecated_alias = False
        if resolved_strategy == "best_hybrid":
            resolved_strategy = "weighted_hybrid_025"
            deprecated_alias = True

        if resolved_strategy == "heuristic":
            return self._heuristic_response(
                request=request,
                requested_strategy="heuristic",
                deprecated_alias=deprecated_alias,
            )
        if resolved_strategy == "deepseek_direct":
            return self._predict_deepseek_direct(
                request=request,
                requested_strategy="deepseek_direct",
                deprecated_alias=deprecated_alias,
            )
        if resolved_strategy == "weighted_hybrid_025":
            return self._experimental_response(
                request=request,
                requested_strategy="weighted_hybrid_025",
                predictor=self.weighted_hybrid_025,
                output_type="experimental_hybrid",
                deprecated_alias=deprecated_alias,
            )
        if resolved_strategy == "learned_hybrid":
            if self.learned_hybrid is None:
                raise RuntimeError("learned_hybrid requested but no combiner artifact was loaded")
            return self._experimental_response(
                request=request,
                requested_strategy="learned_hybrid",
                predictor=self.learned_hybrid,
                output_type="experimental_learned_hybrid",
                deprecated_alias=deprecated_alias,
            )
        raise ValueError(f"Unsupported strategy: {resolved_strategy}")
