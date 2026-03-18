from __future__ import annotations

from dataclasses import dataclass

from spbce.schema.api import PredictBehaviorRequest


@dataclass(slots=True)
class BehaviorPrediction:
    predicted_outcome: float | None
    uncertainty: float
    support_notes: list[str]


class BehaviorModel:
    def predict(self, request: PredictBehaviorRequest) -> BehaviorPrediction:
        return BehaviorPrediction(
            predicted_outcome=None,
            uncertainty=1.0,
            support_notes=[
                "Behavior model is scaffolded but not trained.",
                "Paired survey-behavior data is required for valid outcome forecasting.",
            ],
        )
