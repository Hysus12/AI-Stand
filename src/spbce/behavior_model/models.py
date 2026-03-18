from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import pandas as pd
from sklearn.linear_model import Ridge


@dataclass(slots=True)
class BehaviorOutcomeModel:
    alpha: float = 1.0
    regressor: Ridge = field(init=False, repr=False)
    feature_columns: list[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.regressor = Ridge(alpha=self.alpha)

    def fit(self, frame: pd.DataFrame) -> BehaviorOutcomeModel:
        training_frame = pd.get_dummies(
            frame.drop(columns=["record_id", "group_id", "outcome_value"]),
            dtype=float,
        ).fillna(0.0)
        self.feature_columns = list(training_frame.columns)
        self.regressor.fit(training_frame, frame["outcome_value"].astype(float))
        return self

    def predict(self, frame: pd.DataFrame) -> list[float]:
        scoring_frame = pd.get_dummies(
            frame.drop(columns=["record_id", "group_id", "outcome_value"]),
            dtype=float,
        ).fillna(0.0)
        aligned = scoring_frame.reindex(columns=self.feature_columns, fill_value=0.0)
        predictions = self.regressor.predict(aligned)
        return [float(max(0.0, min(1.0, prediction))) for prediction in predictions]

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> BehaviorOutcomeModel:
        return joblib.load(path)
