from __future__ import annotations

import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from spbce.baselines.direct_probability_llm import DirectProbabilityLlmBaseline
from spbce.baselines.hybrid import HybridPredictor, distribution_entropy
from spbce.baselines.learned_combiner import LearnedHybridCombiner
from spbce.baselines.prompt_only import PromptOnlyPersonaBaseline
from spbce.inference.export import export_project_run
from spbce.inference.generation import generate_synthetic_respondents
from spbce.inference.summary import (
    build_recommendations,
    compute_question_insights,
    render_executive_summary,
)
from spbce.schema.api import (
    MvpCostLatencySummary,
    MvpPredictSurveyResponse,
    PredictSurveyRequest,
    SurveyContext,
)
from spbce.schema.canonical import PopulationStruct
from spbce.schema.project import (
    AggregateSignals,
    ApiUsageDiagnostics,
    FallbackSummary,
    LatencyDiagnostics,
    OutputQualityDiagnostics,
    ProjectDiagnostics,
    ProjectInput,
    ProjectOutput,
    ProjectRunResult,
    QuestionResult,
    SegmentResult,
    SurveyQuestion,
    VariantResult,
    VariantSpec,
)
from spbce.settings import get_provider_environment_summary, initialize_runtime_env


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


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, quantile)) * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return float(
        ordered[lower_index] + ((ordered[upper_index] - ordered[lower_index]) * fraction)
    )


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
    disable_remote_llm: bool = False
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
        context: SurveyContext | dict[str, Any] | None = None,
    ) -> PredictSurveyRequest:
        resolved_context = (
            context
            if isinstance(context, SurveyContext)
            else SurveyContext.model_validate(context or {})
        )
        return PredictSurveyRequest(
            question_text=question_text,
            options=options,
            population_text=population_text,
            population_struct=PopulationStruct.model_validate(population_struct or {}),
            context=resolved_context,
        )

    def load_project_input(self, input_path: str | Path) -> ProjectInput:
        path = Path(input_path)
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return ProjectInput.model_validate(payload)

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
                "sample_attempts": int((attempted_result or {}).get("total_sample_attempts", 0)),
                "final_text_present_count": int(
                    (attempted_result or {}).get("final_text_present_count", 0)
                ),
                "invalid_output_count": int(
                    (attempted_result or {}).get("invalid_output_count", 0)
                ),
                "json_compliance_count": int(
                    (attempted_result or {}).get("json_compliance_count", 0)
                ),
                "parser_failure_count": int(
                    (attempted_result or {}).get("parser_failure_count", 0)
                ),
                "request_latencies_ms": list(
                    (attempted_result or {}).get("request_latencies_ms", [])
                ),
                "usage_source_breakdown": dict(
                    (attempted_result or {}).get("usage_source_breakdown", {})
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

    def _deepseek_provider_available(self) -> bool:
        if self.disable_remote_llm or os.getenv("SPBCE_DISABLE_REMOTE_LLM") == "1":
            return False
        initialize_runtime_env(self.env_file)
        summary = get_provider_environment_summary(self.env_file)
        return "deepseek" in summary.get("providers", [])

    def _predict_deepseek_direct(
        self,
        request: PredictSurveyRequest,
        requested_strategy: str,
        deprecated_alias: bool = False,
    ) -> MvpPredictSurveyResponse:
        if not self._deepseek_provider_available():
            return self._heuristic_response(
                request=request,
                requested_strategy=requested_strategy,
                fallback_reason="provider_unavailable",
                deprecated_alias=deprecated_alias,
                retry_count=0,
            )
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
                        "sample_attempts": int(result.get("total_sample_attempts", 0)),
                        "final_text_present_count": int(
                            result.get("final_text_present_count", 0)
                        ),
                        "invalid_output_count": int(result.get("invalid_output_count", 0)),
                        "json_compliance_count": int(result.get("json_compliance_count", 0)),
                        "parser_failure_count": int(result.get("parser_failure_count", 0)),
                        "request_latencies_ms": list(result.get("request_latencies_ms", [])),
                        "usage_source_breakdown": dict(
                            result.get("usage_source_breakdown", {})
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

    def _effective_variants(self, project: ProjectInput) -> list[VariantSpec | None]:
        return project.variants or [None]

    def _population_text(self, project: ProjectInput, segment_id: str) -> str:
        segment = next(
            candidate for candidate in project.target_segments if candidate.segment_id == segment_id
        )
        category_suffix = f" in category {project.category}" if project.category else ""
        return (
            f"Target segment for {project.product_name}{category_suffix}: "
            f"{segment.segment_name}. {segment.combined_description()}"
        )

    def _project_question_text(
        self,
        project: ProjectInput,
        question: SurveyQuestion,
        variant: VariantSpec | None,
    ) -> str:
        lines = [f"Product: {project.product_name}", f"Brief: {project.product_brief.strip()}"]
        if project.category:
            lines.append(f"Category: {project.category}")
        if project.category_hints:
            lines.append(f"Category hints: {', '.join(project.category_hints)}")
        if variant is not None:
            lines.append(f"Variant: {variant.variant_name}")
            if variant.variant_description:
                lines.append(f"Variant description: {variant.variant_description}")
            if variant.message:
                lines.append(f"Message: {variant.message}")
            if variant.feature:
                lines.append(f"Feature focus: {variant.feature}")
            if variant.price is not None:
                lines.append(f"Price: {variant.price}")
        if question.theme:
            lines.append(f"Theme: {question.theme}")
        if question.tags:
            lines.append(f"Tags: {', '.join(question.tags)}")
        lines.append(f"Survey question: {question.question_text}")
        return "\n".join(lines)

    def _question_score(
        self,
        question: SurveyQuestion,
        distribution: list[float],
    ) -> float | None:
        if question.scoring_direction == "neutral":
            return None
        if len(distribution) == 1:
            return float(distribution[0])
        utilities = [
            1.0 - (index / (len(distribution) - 1)) for index in range(len(distribution))
        ]
        if question.scoring_direction == "positive_low":
            utilities = list(reversed(utilities))
        return float(
            sum(
                probability * utility
                for probability, utility in zip(distribution, utilities, strict=True)
            )
        )

    def _aggregate_signals(self, question_results: list[QuestionResult]) -> AggregateSignals:
        scores = [
            (
                float(question_result.normalized_score),
                float(question_result.diagnostics.get("decision_weight", 1.0)),
            )
            for question_result in question_results
            if question_result.normalized_score is not None
        ]
        total_weight = sum(weight for _, weight in scores)
        weighted_score = (
            float(sum(score * weight for score, weight in scores) / total_weight)
            if total_weight > 0
            else None
        )
        confidence_values = [question_result.confidence for question_result in question_results]
        uncertainty_values = [question_result.uncertainty for question_result in question_results]
        margin_values = [
            float(question_result.diagnostics.get("top1_margin", 0.0))
            for question_result in question_results
        ]
        return AggregateSignals(
            weighted_score=weighted_score,
            mean_confidence=float(sum(confidence_values) / len(confidence_values))
            if confidence_values
            else 0.0,
            mean_uncertainty=float(sum(uncertainty_values) / len(uncertainty_values))
            if uncertainty_values
            else 0.0,
            fallback_rate=float(
                sum(1 for result in question_results if result.fallback_happened)
                / len(question_results)
            )
            if question_results
            else 0.0,
            invalid_rate=float(
                sum(1 for result in question_results if result.diagnostics.get("invalid_flag"))
                / len(question_results)
            )
            if question_results
            else 0.0,
            mean_top_option_margin=float(sum(margin_values) / len(margin_values))
            if margin_values
            else 0.0,
            question_count=len(question_results),
        )

    def _variant_context(
        self,
        project: ProjectInput,
        question: SurveyQuestion,
        variant: VariantSpec | None,
    ) -> SurveyContext:
        return SurveyContext(
            product_category=project.category,
            price=variant.price if variant is not None else None,
            market_metadata={
                "product_name": project.product_name,
                "product_brief": project.product_brief,
                "category_hints": project.category_hints,
                "variant_id": variant.variant_id if variant is not None else None,
                "variant_name": variant.variant_name if variant is not None else None,
                "variant_description": variant.variant_description if variant is not None else None,
                "question_id": question.question_id,
                "question_type": question.question_type,
                "question_tags": question.tags,
            },
        )

    def _attempt_metrics_from_response(
        self,
        response: MvpPredictSurveyResponse,
    ) -> dict[str, Any]:
        metadata = response.metadata or {}
        sample_attempts = int(metadata.get("sample_attempts", 0))
        return {
            "sample_attempts": sample_attempts,
            "final_text_present_count": int(metadata.get("final_text_present_count", 0)),
            "invalid_output_count": int(metadata.get("invalid_output_count", 0)),
            "json_compliance_count": int(metadata.get("json_compliance_count", 0)),
            "parser_failure_count": int(metadata.get("parser_failure_count", 0)),
            "request_latencies_ms": [
                float(value) for value in metadata.get("request_latencies_ms", [])
            ],
            "usage_source_breakdown": dict(metadata.get("usage_source_breakdown", {})),
            "total_input_tokens": int(response.cost_latency_summary.total_input_tokens),
            "total_output_tokens": int(response.cost_latency_summary.total_output_tokens),
            "estimated_api_cost_usd": float(response.cost_latency_summary.estimated_api_cost_usd),
            "total_latency_ms": float(response.cost_latency_summary.total_latency_ms),
            "request_count": int(response.cost_latency_summary.request_count),
        }

    def _build_project_diagnostics(
        self,
        all_question_results: list[QuestionResult],
        question_attempt_records: list[dict[str, Any]],
        fallback_summary: FallbackSummary,
        total_runtime_seconds: float,
    ) -> ProjectDiagnostics:
        latencies_ms = [
            latency_ms
            for record in question_attempt_records
            for latency_ms in record["request_latencies_ms"]
        ]
        total_input_tokens = sum(record["total_input_tokens"] for record in question_attempt_records)
        total_output_tokens = sum(
            record["total_output_tokens"] for record in question_attempt_records
        )
        estimated_api_cost_usd = float(
            sum(record["estimated_api_cost_usd"] for record in question_attempt_records)
        )
        total_sample_attempts = sum(record["sample_attempts"] for record in question_attempt_records)
        final_text_present_count = sum(
            record["final_text_present_count"] for record in question_attempt_records
        )
        invalid_output_count = sum(
            record["invalid_output_count"] for record in question_attempt_records
        )
        parser_failure_count = sum(
            record["parser_failure_count"] for record in question_attempt_records
        )
        json_compliance_count = sum(
            record["json_compliance_count"] for record in question_attempt_records
        )
        direct_success_count = sum(
            1
            for question_result in all_question_results
            if question_result.actual_strategy_used == "deepseek_direct"
            and not question_result.fallback_happened
        )
        total_predictions = len(all_question_results)
        usage_sources: Counter[str] = Counter()
        for record in question_attempt_records:
            usage_sources.update(record["usage_source_breakdown"])
        usage_notes: list[str] = []
        if usage_sources:
            source_labels = ", ".join(
                f"{source}={count}" for source, count in sorted(usage_sources.items())
            )
            usage_notes.append(f"Usage token source counts: {source_labels}.")
        else:
            usage_notes.append(
                "Provider usage metadata was unavailable for one or more requests; token totals may be incomplete."
            )

        fallback_events = [
            {
                "segment_id": question_result.diagnostics.get("segment_id"),
                "segment_name": question_result.diagnostics.get("segment_name"),
                "variant_id": question_result.diagnostics.get("variant_id"),
                "variant_name": question_result.diagnostics.get("variant_name"),
                "question_id": question_result.question_id,
                "question_text": question_result.question_text,
                "fallback_reason": question_result.fallback_reason,
                "actual_strategy_used": question_result.actual_strategy_used,
                "estimated_api_cost_usd": float(
                    question_result.diagnostics.get("cost_latency_summary", {}).get(
                        "estimated_api_cost_usd", 0.0
                    )
                ),
            }
            for question_result in all_question_results
            if question_result.fallback_happened
        ]
        slowest_requests = sorted(
            question_attempt_records,
            key=lambda record: (record["total_latency_ms"], record["question_id"]),
            reverse=True,
        )[:5]

        return ProjectDiagnostics(
            mean_confidence=float(
                sum(question_result.confidence for question_result in all_question_results)
                / len(all_question_results)
            )
            if all_question_results
            else 0.0,
            mean_uncertainty=float(
                sum(question_result.uncertainty for question_result in all_question_results)
                / len(all_question_results)
            )
            if all_question_results
            else 0.0,
            invalid_rate=float(
                sum(
                    1
                    for question_result in all_question_results
                    if question_result.diagnostics.get("invalid_flag")
                )
                / len(all_question_results)
            )
            if all_question_results
            else 0.0,
            api_usage=ApiUsageDiagnostics(
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                estimated_api_cost_usd=estimated_api_cost_usd,
                estimated_cost_per_valid_question_usd=float(
                    estimated_api_cost_usd / direct_success_count
                )
                if direct_success_count
                else 0.0,
                usage_notes=usage_notes,
            ),
            latency=LatencyDiagnostics(
                total_runtime_seconds=float(total_runtime_seconds),
                total_request_count=len(latencies_ms),
                average_latency_ms_per_request=float(sum(latencies_ms) / len(latencies_ms))
                if latencies_ms
                else 0.0,
                p50_latency_ms=percentile(latencies_ms, 0.50),
                p95_latency_ms=percentile(latencies_ms, 0.95),
                slowest_requests=[
                    {
                        "segment_id": record["segment_id"],
                        "segment_name": record["segment_name"],
                        "variant_id": record["variant_id"],
                        "variant_name": record["variant_name"],
                        "question_id": record["question_id"],
                        "question_text": record["question_text"],
                        "total_latency_ms": float(record["total_latency_ms"]),
                        "request_count": int(record["request_count"]),
                        "average_latency_ms_per_request": float(
                            record["total_latency_ms"] / record["request_count"]
                        )
                        if record["request_count"]
                        else 0.0,
                        "fallback_happened": bool(record["fallback_happened"]),
                        "fallback_reason": record["fallback_reason"],
                        "actual_strategy_used": record["actual_strategy_used"],
                    }
                    for record in slowest_requests
                ],
            ),
            output_quality=OutputQualityDiagnostics(
                total_predictions=total_predictions,
                direct_success_count=direct_success_count,
                direct_success_rate=float(direct_success_count / total_predictions)
                if total_predictions
                else 0.0,
                final_text_present_rate=float(final_text_present_count / total_sample_attempts)
                if total_sample_attempts
                else 0.0,
                invalid_output_rate=float(invalid_output_count / total_sample_attempts)
                if total_sample_attempts
                else 0.0,
                parse_failure_rate=float(parser_failure_count / total_sample_attempts)
                if total_sample_attempts
                else 0.0,
                response_schema_compliance=float(json_compliance_count / total_sample_attempts)
                if total_sample_attempts
                else 0.0,
                fallback_rate=fallback_summary.fallback_rate,
                fallback_events=fallback_events,
            ),
            limitations=[
                (
                    "Pilot MVP only supports 7 to 15 closed-ended questions and up to "
                    "3 segments / 3 variants."
                ),
                (
                    "Synthetic respondents use a lightweight latent-profile sampler "
                    "rather than a learned joint respondent model."
                ),
                (
                    "Behavior validity has not been validated on customer-specific "
                    "commercial outcomes; do not claim superiority over human surveys."
                ),
                (
                    "Routing / skip logic and long open-ended questionnaires are "
                    "intentionally out of scope."
                ),
            ],
        )

    def run_project(self, project_input: ProjectInput | dict[str, Any]) -> ProjectRunResult:
        project = (
            project_input
            if isinstance(project_input, ProjectInput)
            else ProjectInput.model_validate(project_input)
        )
        original_retry_attempts = self.max_retry_attempts
        original_fallback_min_confidence = self.fallback_min_confidence
        self.max_retry_attempts = project.inference_settings.max_retry_attempts
        self.fallback_min_confidence = project.inference_settings.fallback_min_confidence

        try:
            run_started = time.perf_counter()
            fallback_reason_counter: Counter[str] = Counter()
            actual_strategy_counter: Counter[str] = Counter()
            all_question_results: list[QuestionResult] = []
            question_attempt_records: list[dict[str, Any]] = []
            segment_results: list[SegmentResult] = []

            for segment in project.target_segments:
                variant_results: list[VariantResult] = []
                for variant in self._effective_variants(project):
                    question_results: list[QuestionResult] = []
                    for question in project.survey_questions:
                        request = self.build_request(
                            question_text=self._project_question_text(project, question, variant),
                            options=question.options,
                            population_text=self._population_text(project, segment.segment_id),
                            population_struct=segment.population_struct.model_dump(mode="python"),
                            context=self._variant_context(project, question, variant),
                        )
                        response = self.predict(
                            request,
                            strategy=project.inference_settings.primary_strategy,
                        )
                        ordered_distribution = [
                            float(response.distribution[option]) for option in question.options
                        ]
                        confidence_summary = response.metadata.get("confidence_summary", {})
                        question_result = QuestionResult(
                            question_id=question.question_id,
                            question_text=question.question_text,
                            question_type=question.question_type,
                            options=question.options,
                            distribution=response.distribution,
                            top_option=response.top_option,
                            confidence=response.confidence,
                            uncertainty=response.uncertainty,
                            normalized_score=self._question_score(question, ordered_distribution),
                            requested_strategy=response.requested_strategy,
                            actual_strategy_used=response.actual_strategy_used,
                            fallback_happened=response.fallback_happened,
                            fallback_reason=response.fallback_reason,
                            tags=question.tags,
                            diagnostics={
                                "invalid_flag": response.invalid_flag,
                                "segment_id": segment.segment_id,
                                "segment_name": segment.segment_name,
                                "variant_id": variant.variant_id if variant is not None else None,
                                "variant_name": (
                                    variant.variant_name if variant is not None else "base"
                                ),
                                "decision_weight": question.decision_weight,
                                "scoring_direction": question.scoring_direction,
                                "theme": question.theme,
                                "top1_margin": confidence_summary.get(
                                    "top1_margin",
                                    distribution_margin(ordered_distribution),
                                ),
                                "entropy_normalized": confidence_summary.get(
                                    "entropy_normalized",
                                    distribution_entropy(ordered_distribution),
                                ),
                                "cost_latency_summary": response.cost_latency_summary.model_dump(
                                    mode="json"
                                ),
                                "metadata": response.metadata,
                            },
                        )
                        question_results.append(question_result)
                        all_question_results.append(question_result)
                        attempt_metrics = self._attempt_metrics_from_response(response)
                        question_attempt_records.append(
                            {
                                "segment_id": segment.segment_id,
                                "segment_name": segment.segment_name,
                                "variant_id": variant.variant_id if variant is not None else None,
                                "variant_name": (
                                    variant.variant_name if variant is not None else "base"
                                ),
                                "question_id": question.question_id,
                                "question_text": question.question_text,
                                "fallback_happened": bool(response.fallback_happened),
                                "fallback_reason": response.fallback_reason,
                                "actual_strategy_used": response.actual_strategy_used,
                                **attempt_metrics,
                            }
                        )
                        actual_strategy_counter[response.actual_strategy_used] += 1
                        if response.fallback_happened:
                            fallback_reason_counter[response.fallback_reason or "unspecified"] += 1

                    aggregate_signals = self._aggregate_signals(question_results)
                    variant_results.append(
                        VariantResult(
                            variant_id=variant.variant_id if variant is not None else None,
                            variant_name=variant.variant_name if variant is not None else "base",
                            variant_description=(
                                variant.variant_description if variant is not None else None
                            ),
                            question_results=question_results,
                            aggregate_signals=aggregate_signals,
                            diagnostics={
                                "question_count": len(question_results),
                                "fallback_rate": aggregate_signals.fallback_rate,
                            },
                        )
                    )

                segment_aggregate = self._aggregate_signals(
                    [
                        question_result
                        for variant_result in variant_results
                        for question_result in variant_result.question_results
                    ]
                )
                segment_results.append(
                    SegmentResult(
                        segment_id=segment.segment_id,
                        segment_name=segment.segment_name,
                        demographic_description=segment.demographic_description,
                        psychographic_description=segment.psychographic_description,
                        behavioral_description=segment.behavioral_description,
                        variant_results=variant_results,
                        aggregate_signals=segment_aggregate,
                        diagnostics={
                            "variant_count": len(variant_results),
                            "estimated_weight": segment.estimated_weight,
                        },
                    )
                )

            fallback_count = sum(fallback_reason_counter.values())
            fallback_summary = FallbackSummary(
                total_predictions=len(all_question_results),
                fallback_count=fallback_count,
                fallback_rate=float(fallback_count / len(all_question_results))
                if all_question_results
                else 0.0,
                reason_breakdown=dict(sorted(fallback_reason_counter.items())),
                actual_strategy_breakdown=dict(sorted(actual_strategy_counter.items())),
            )

            diagnostics = self._build_project_diagnostics(
                all_question_results=all_question_results,
                question_attempt_records=question_attempt_records,
                fallback_summary=fallback_summary,
                total_runtime_seconds=(time.perf_counter() - run_started),
            )
            project_output = ProjectOutput(
                project_id=project.project_id,
                product_name=project.product_name,
                model_route_used="deepseek_direct -> heuristic fallback",
                fallback_summary=fallback_summary,
                per_segment_results=segment_results,
                executive_summary="",
                recommendations=[],
                diagnostics=diagnostics,
            )

            question_insights = compute_question_insights(project_output)
            project_output.diagnostics.question_level_results = question_insights
            project_output.recommendations = build_recommendations(
                project_output,
                question_insights,
            )
            project_output.executive_summary = render_executive_summary(
                project_output,
                question_insights,
                project_output.recommendations,
            )
            synthetic_respondents, generator_notes = generate_synthetic_respondents(
                project=project,
                output=project_output,
                respondent_count=project.generation_settings.synthetic_respondent_count,
            )
            project_output.diagnostics.generator_notes = generator_notes
            return ProjectRunResult(
                project_output=project_output,
                synthetic_respondents=synthetic_respondents,
            )
        finally:
            self.max_retry_attempts = original_retry_attempts
            self.fallback_min_confidence = original_fallback_min_confidence

    def run_project_to_directory(
        self,
        project_input: ProjectInput | dict[str, Any],
        output_dir: str | Path,
        synthetic_respondent_count: int | None = None,
    ) -> ProjectRunResult:
        project = (
            project_input
            if isinstance(project_input, ProjectInput)
            else ProjectInput.model_validate(project_input)
        )
        if synthetic_respondent_count is not None:
            project = project.model_copy(
                update={
                    "generation_settings": project.generation_settings.model_copy(
                        update={"synthetic_respondent_count": synthetic_respondent_count}
                    )
                }
            )
        run_result = self.run_project(project)
        export_project_run(run_result, output_dir=output_dir)
        return run_result

    def run_project_file(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        synthetic_respondent_count: int | None = None,
    ) -> ProjectRunResult:
        return self.run_project_to_directory(
            project_input=self.load_project_input(input_path),
            output_dir=output_dir,
            synthetic_respondent_count=synthetic_respondent_count,
        )
