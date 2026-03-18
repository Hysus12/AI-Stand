from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from spbce.baselines.direct_probability_llm import DirectProbabilityLlmBaseline
from spbce.baselines.hybrid import distribution_entropy
from spbce.baselines.prompt_only import PromptOnlyPersonaBaseline
from spbce.metrics.distributions import js_divergence, normalize_distribution
from spbce.schema.api import PredictSurveyRequest
from spbce.utils.text import simple_tokenize


def distribution_margin(distribution: list[float]) -> float:
    if len(distribution) < 2:
        return float(distribution[0]) if distribution else 0.0
    ordered = sorted(distribution, reverse=True)
    return float(ordered[0] - ordered[1])


def pad_sorted_distribution(distribution: list[float], size: int = 7) -> list[float]:
    padded = sorted(distribution, reverse=True)[:size]
    padded.extend([0.0] * max(0, size - len(padded)))
    return [float(value) for value in padded]


def optimal_mixing_weight(
    heuristic_distribution: list[float],
    llm_distribution: list[float],
    observed_distribution: list[float],
    step: float = 0.05,
) -> float:
    weights = np.arange(0.0, 1.0 + step, step)
    best_weight = 0.0
    best_score = float("inf")
    heuristic = np.asarray(heuristic_distribution, dtype=float)
    llm = np.asarray(llm_distribution, dtype=float)
    observed = np.asarray(observed_distribution, dtype=float)
    for weight in weights:
        blended = normalize_distribution(((1.0 - weight) * heuristic) + (weight * llm))
        score = js_divergence(blended.tolist(), observed.tolist())
        if score < best_score:
            best_score = score
            best_weight = float(weight)
    return best_weight


@dataclass(slots=True)
class LearnedHybridCombiner:
    heuristic_predictor: PromptOnlyPersonaBaseline
    llm_predictor: DirectProbabilityLlmBaseline
    feature_names: list[str] = field(default_factory=list)
    model: Ridge | None = None
    scaler: StandardScaler | None = None
    training_metadata: dict[str, Any] = field(default_factory=dict)

    def _build_feature_row(
        self,
        request: PredictSurveyRequest,
        heuristic_distribution: list[float],
        llm_result: dict[str, Any],
    ) -> tuple[list[float], dict[str, float]]:
        llm_distribution = llm_result["distribution"] or normalize_distribution(
            np.ones(len(request.options), dtype=float)
        ).tolist()
        features: dict[str, float] = {
            "option_count": float(len(request.options)),
            "question_token_count": float(len(simple_tokenize(request.question_text))),
            "population_token_count": float(len(simple_tokenize(request.population_text))),
            "heuristic_top1": float(max(heuristic_distribution)),
            "heuristic_entropy": distribution_entropy(heuristic_distribution),
            "heuristic_margin": distribution_margin(heuristic_distribution),
            "llm_top1": float(max(llm_distribution)),
            "llm_entropy": distribution_entropy(llm_distribution),
            "llm_margin": distribution_margin(llm_distribution),
            "llm_json_compliance_rate": float(llm_result.get("json_compliance_rate", 0.0)),
            "llm_invalid_output_rate": float(llm_result.get("invalid_output_rate", 0.0)),
            "llm_final_text_present_rate": float(llm_result.get("final_text_present_rate", 0.0)),
            "heuristic_llm_js_gap": js_divergence(heuristic_distribution, llm_distribution),
            "heuristic_llm_l1_gap": float(
                np.abs(
                    np.asarray(heuristic_distribution, dtype=float)
                    - np.asarray(llm_distribution, dtype=float)
                ).sum()
            ),
        }
        for index, value in enumerate(pad_sorted_distribution(heuristic_distribution), start=1):
            features[f"heuristic_prob_rank_{index}"] = value
        for index, value in enumerate(pad_sorted_distribution(llm_distribution), start=1):
            features[f"llm_prob_rank_{index}"] = value
        ordered_feature_names = self.feature_names or sorted(features)
        return [features[name] for name in ordered_feature_names], features

    def fit_rows(self, rows: list[dict[str, Any]]) -> LearnedHybridCombiner:
        if not rows:
            raise ValueError("rows cannot be empty")
        self.feature_names = sorted(rows[0]["features"])
        matrix = np.asarray(
            [[float(row["features"][name]) for name in self.feature_names] for row in rows],
            dtype=float,
        )
        targets = np.asarray([float(row["target_weight"]) for row in rows], dtype=float)
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(matrix)
        self.model = Ridge(alpha=1.0, random_state=42)
        self.model.fit(scaled, targets)
        return self

    def predict_weight_from_result(
        self,
        request: PredictSurveyRequest,
        heuristic_distribution: list[float],
        llm_result: dict[str, Any],
    ) -> tuple[float, dict[str, float]]:
        if self.model is None or self.scaler is None:
            raise RuntimeError("combiner model is not fitted")
        feature_row, feature_dict = self._build_feature_row(
            request=request,
            heuristic_distribution=heuristic_distribution,
            llm_result=llm_result,
        )
        weight = float(self.model.predict(self.scaler.transform([feature_row]))[0])
        return float(min(1.0, max(0.0, weight))), feature_dict

    def sample_distribution(
        self, request: PredictSurveyRequest, few_shot: bool = False
    ) -> dict[str, Any]:
        del few_shot
        heuristic_distribution = self.heuristic_predictor.predict_proba(request)
        llm_result = self.llm_predictor.sample_distribution(request, few_shot=False)
        llm_distribution = llm_result["distribution"]
        if llm_distribution is None:
            return {
                **llm_result,
                "distribution": heuristic_distribution,
                "scorable": True,
                "hybrid_decision": "heuristic_fallback",
                "hybrid_fallback_used": True,
                "predicted_llm_weight": 0.0,
            }
        weight, feature_dict = self.predict_weight_from_result(
            request=request,
            heuristic_distribution=heuristic_distribution,
            llm_result=llm_result,
        )
        blended = normalize_distribution(
            ((1.0 - weight) * np.asarray(heuristic_distribution, dtype=float))
            + (weight * np.asarray(llm_distribution, dtype=float))
        ).tolist()
        return {
            **llm_result,
            "distribution": blended,
            "scorable": True,
            "hybrid_decision": "learned_weighted_blend",
            "hybrid_fallback_used": False,
            "predicted_llm_weight": weight,
            "feature_snapshot": feature_dict,
            "generation_config": self.generation_config(),
        }

    def generation_config(self) -> dict[str, Any]:
        return {
            "strategy": "learned_combiner_v1",
            "feature_names": self.feature_names,
            "training_metadata": self.training_metadata,
            "llm_generation_config": self.llm_predictor.generation_config(),
        }

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {
                "feature_names": self.feature_names,
                "model": self.model,
                "scaler": self.scaler,
                "training_metadata": self.training_metadata,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        heuristic_predictor: PromptOnlyPersonaBaseline,
        llm_predictor: DirectProbabilityLlmBaseline,
    ) -> LearnedHybridCombiner:
        payload = joblib.load(path)
        return cls(
            heuristic_predictor=heuristic_predictor,
            llm_predictor=llm_predictor,
            feature_names=list(payload["feature_names"]),
            model=payload["model"],
            scaler=payload["scaler"],
            training_metadata=dict(payload.get("training_metadata", {})),
        )
