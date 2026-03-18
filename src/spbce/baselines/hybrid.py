from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import log
from typing import Any

from spbce.baselines.direct_probability_llm import DirectProbabilityLlmBaseline
from spbce.baselines.prompt_only import PromptOnlyPersonaBaseline
from spbce.metrics.distributions import normalize_distribution
from spbce.schema.api import PredictSurveyRequest


def distribution_entropy(distribution: list[float]) -> float:
    if not distribution:
        return 0.0
    entropy = 0.0
    for probability in distribution:
        if probability > 0:
            entropy -= probability * log(probability)
    max_entropy = log(len(distribution)) if len(distribution) > 1 else 1.0
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


@dataclass(slots=True)
class HybridPredictor:
    name: str
    heuristic_predictor: PromptOnlyPersonaBaseline
    llm_predictor: DirectProbabilityLlmBaseline
    strategy: str
    config: dict[str, Any]
    decision_counter: Counter[str] = field(default_factory=Counter, init=False)

    def generation_config(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "config": self.config,
            "llm_generation_config": self.llm_predictor.generation_config(),
        }

    def _weighted_average(
        self, heuristic_distribution: list[float], llm_distribution: list[float], llm_weight: float
    ) -> list[float]:
        heuristic_weight = 1.0 - llm_weight
        scores = [
            (heuristic_weight * heuristic_probability) + (llm_weight * llm_probability)
            for heuristic_probability, llm_probability in zip(
                heuristic_distribution, llm_distribution, strict=True
            )
        ]
        return normalize_distribution(scores).tolist()

    def sample_distribution(
        self, request: PredictSurveyRequest, few_shot: bool = False
    ) -> dict[str, Any]:
        del few_shot
        heuristic_distribution = self.heuristic_predictor.predict_proba(request)
        llm_result = self.llm_predictor.sample_distribution(request, few_shot=False)
        llm_distribution = llm_result["distribution"]
        llm_valid = bool(llm_result["scorable"]) and llm_distribution is not None
        llm_top1 = float(max(llm_distribution)) if llm_distribution is not None else 0.0
        llm_entropy = distribution_entropy(llm_distribution or [])

        final_distribution = heuristic_distribution
        decision = "heuristic_default"
        fallback_used = False

        if self.strategy == "weighted_average":
            llm_weight = float(self.config["llm_weight"]) if llm_valid else 0.0
            final_distribution = (
                self._weighted_average(heuristic_distribution, llm_distribution, llm_weight)
                if llm_valid and llm_distribution is not None
                else heuristic_distribution
            )
            decision = f"weighted_average_{llm_weight:.2f}"
            fallback_used = not llm_valid
        elif self.strategy == "confidence_gated":
            min_top1 = float(self.config.get("min_top1_probability", 0.55))
            max_entropy = float(self.config.get("max_entropy", 0.80))
            llm_weight_if_pass = float(self.config.get("llm_weight_if_pass", 0.75))
            require_json = bool(self.config.get("require_json_compliance", True))
            require_invalid_zero = bool(self.config.get("require_invalid_zero", True))
            passes = llm_valid
            passes = passes and (not require_json or llm_result["json_compliance_rate"] >= 1.0)
            passes = passes and (
                not require_invalid_zero or float(llm_result["invalid_output_rate"]) == 0.0
            )
            passes = passes and llm_top1 >= min_top1
            passes = passes and llm_entropy <= max_entropy
            if passes and llm_distribution is not None:
                final_distribution = self._weighted_average(
                    heuristic_distribution,
                    llm_distribution,
                    llm_weight_if_pass,
                )
                decision = "gated_llm_weighted"
            else:
                final_distribution = heuristic_distribution
                decision = "gated_heuristic"
                fallback_used = True
        elif self.strategy == "mixture_switch":
            min_llm_top1 = float(self.config.get("min_llm_top1_probability", 0.60))
            max_llm_entropy = float(self.config.get("max_llm_entropy", 0.78))
            min_option_count_for_llm = int(self.config.get("min_option_count_for_llm", 3))
            blend_weight = float(self.config.get("blend_llm_weight", 0.5))
            if (
                llm_valid
                and llm_distribution is not None
                and len(request.options) >= min_option_count_for_llm
                and llm_top1 >= min_llm_top1
                and llm_entropy <= max_llm_entropy
            ):
                final_distribution = llm_distribution
                decision = "switch_llm"
            elif llm_valid and llm_distribution is not None:
                final_distribution = self._weighted_average(
                    heuristic_distribution,
                    llm_distribution,
                    blend_weight,
                )
                decision = "switch_blend"
            else:
                final_distribution = heuristic_distribution
                decision = "switch_heuristic"
                fallback_used = True
        else:  # pragma: no cover
            raise ValueError(f"Unsupported hybrid strategy: {self.strategy}")

        self.decision_counter[decision] += 1
        return {
            **llm_result,
            "distribution": final_distribution,
            "scorable": True,
            "invalid_output_rate": 0.0,
            "parser_failure_rate": 0.0,
            "hybrid_config": self.generation_config(),
            "hybrid_decision": decision,
            "hybrid_fallback_used": fallback_used,
            "llm_branch_valid": llm_valid,
            "llm_branch_top1_probability": llm_top1,
            "llm_branch_entropy": llm_entropy,
        }
