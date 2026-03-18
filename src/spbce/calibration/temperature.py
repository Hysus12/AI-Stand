from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spbce.metrics.distributions import EPSILON, normalize_distribution


@dataclass(slots=True)
class TemperatureScaler:
    temperature: float = 1.0

    def fit(
        self, predicted_distributions: list[list[float]], true_distributions: list[list[float]]
    ) -> TemperatureScaler:
        temperatures = np.linspace(0.5, 3.0, 26)
        best_temperature = 1.0
        best_loss = float("inf")
        for candidate in temperatures:
            loss = 0.0
            for predicted, observed in zip(
                predicted_distributions, true_distributions, strict=True
            ):
                calibrated = self.apply(predicted, temperature=float(candidate))
                observed_array = normalize_distribution(observed)
                loss += -float(
                    np.sum(observed_array * np.log(np.asarray(calibrated, dtype=float) + EPSILON))
                )
            if loss < best_loss:
                best_loss = loss
                best_temperature = float(candidate)
        self.temperature = best_temperature
        return self

    def apply(self, distribution: list[float], temperature: float | None = None) -> list[float]:
        current_temperature = temperature or self.temperature
        values = np.asarray(distribution, dtype=float)
        logits = np.log(values + EPSILON) / current_temperature
        shifted = logits - logits.max()
        scores = np.exp(shifted)
        return normalize_distribution(scores).tolist()
