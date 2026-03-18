from __future__ import annotations

from collections import defaultdict

import numpy as np

from spbce.baselines.majority import MajorityDistributionBaseline
from spbce.metrics.distributions import normalize_distribution
from spbce.schema.api import PredictSurveyRequest
from spbce.schema.canonical import SurveyRecord


class SubgroupMarginalBaseline:
    def __init__(self) -> None:
        self.majority = MajorityDistributionBaseline()
        self.lookup: dict[tuple[str, int], np.ndarray] = {}

    def fit(self, records: list[SurveyRecord]) -> SubgroupMarginalBaseline:
        self.majority.fit(records)
        grouped: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
        for record in records:
            key = (record.country or record.population_signature(), len(record.options))
            grouped[key].append(np.asarray(record.observed_distribution, dtype=float))
        self.lookup = {
            key: normalize_distribution(np.mean(distributions, axis=0))
            for key, distributions in grouped.items()
        }
        return self

    def predict_proba(self, request: PredictSurveyRequest) -> list[float]:
        key = (request.population_struct.region or request.population_text, len(request.options))
        distribution = self.lookup.get(key)
        if distribution is None:
            return self.majority.predict_proba(request)
        return normalize_distribution(distribution).tolist()
