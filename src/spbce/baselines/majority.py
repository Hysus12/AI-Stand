from __future__ import annotations

from collections import defaultdict

import numpy as np

from spbce.metrics.distributions import normalize_distribution
from spbce.schema.api import PredictSurveyRequest
from spbce.schema.canonical import SurveyRecord


class MajorityDistributionBaseline:
    def __init__(self) -> None:
        self.by_option_count: dict[int, np.ndarray] = {}

    def fit(self, records: list[SurveyRecord]) -> MajorityDistributionBaseline:
        grouped: dict[int, list[np.ndarray]] = defaultdict(list)
        for record in records:
            grouped[len(record.options)].append(
                np.asarray(record.observed_distribution, dtype=float)
            )
        self.by_option_count = {
            option_count: normalize_distribution(np.mean(distributions, axis=0))
            for option_count, distributions in grouped.items()
        }
        return self

    def predict_proba(self, request: PredictSurveyRequest) -> list[float]:
        distribution = self.by_option_count.get(len(request.options))
        if distribution is None:
            distribution = np.full(len(request.options), 1.0 / len(request.options))
        return normalize_distribution(distribution).tolist()
