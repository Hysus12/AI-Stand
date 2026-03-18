from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spbce.metrics.distributions import normalize_distribution
from spbce.schema.api import PredictSurveyRequest
from spbce.utils.text import simple_tokenize

_hf_pipeline: Any | None
try:
    from transformers import pipeline as _hf_pipeline
except Exception:  # pragma: no cover
    _hf_pipeline = None

hf_pipeline: Any | None = _hf_pipeline


POSITIVE_HINTS = {"support", "approve", "yes", "agree", "important", "trust", "good"}
NEGATIVE_HINTS = {"oppose", "disapprove", "no", "disagree", "unimportant", "bad"}


@dataclass(slots=True)
class PromptOnlyPersonaBaseline:
    backend: str = "heuristic"
    model_name: str = "facebook/bart-large-mnli"
    _classifier: object | None = field(init=False, default=None, repr=False)

    def _lazy_classifier(self):  # type: ignore[no-untyped-def]
        if self.backend != "zero_shot":
            return None
        if self._classifier is None and hf_pipeline is not None:
            self._classifier = hf_pipeline("zero-shot-classification", model=self.model_name)
        return self._classifier

    def _heuristic_distribution(self, request: PredictSurveyRequest) -> list[float]:
        question_tokens = set(simple_tokenize(request.question_text))
        scores: list[float] = []
        for option in request.options:
            option_tokens = set(simple_tokenize(option))
            overlap = len(question_tokens.intersection(option_tokens))
            positive_bonus = 1.0 if option_tokens.intersection(POSITIVE_HINTS) else 0.0
            negative_bonus = 0.5 if option_tokens.intersection(NEGATIVE_HINTS) else 0.0
            prior_bonus = 0.2 if request.population_struct.region else 0.0
            scores.append(1.0 + overlap + positive_bonus + negative_bonus + prior_bonus)
        return normalize_distribution(scores).tolist()

    def predict_proba(self, request: PredictSurveyRequest) -> list[float]:
        classifier = self._lazy_classifier()
        if classifier is None:
            return self._heuristic_distribution(request)
        result = classifier(
            sequences=(
                "You are answering a survey as part of the population "
                f"'{request.population_text}'. Question: {request.question_text}"
            ),
            candidate_labels=request.options,
            multi_label=False,
        )
        return normalize_distribution(list(result["scores"])).tolist()
