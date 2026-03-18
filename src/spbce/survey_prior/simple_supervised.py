from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from spbce.features.survey_features import build_option_rows, build_request_frame
from spbce.metrics.distributions import normalize_distribution
from spbce.schema.api import PredictSurveyRequest
from spbce.schema.canonical import SurveyRecord


@dataclass(slots=True)
class SimpleSupervisedSurveyPrior:
    random_state: int = 42
    text_max_features: int = 8000
    pipeline: Pipeline = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.pipeline = Pipeline(
            steps=[
                (
                    "features",
                    ColumnTransformer(
                        transformers=[
                            (
                                "text",
                                TfidfVectorizer(
                                    max_features=self.text_max_features, ngram_range=(1, 2)
                                ),
                                "combined_text",
                            ),
                            (
                                "categorical",
                                OneHotEncoder(handle_unknown="ignore"),
                                [
                                    "population_signature",
                                    "domain",
                                    "country",
                                    "question_topic",
                                    "question_type",
                                    "option_index",
                                ],
                            ),
                        ]
                    ),
                ),
                ("regressor", Ridge(alpha=1.0, random_state=self.random_state)),
            ]
        )

    def fit(self, records: list[SurveyRecord]) -> SimpleSupervisedSurveyPrior:
        frame = build_option_rows(records)
        x_frame = frame.drop(columns=["target_probability"])
        y = frame["target_probability"].to_numpy(dtype=float)
        self.pipeline.fit(x_frame, y)
        return self

    def predict_scores(self, frame: pd.DataFrame) -> np.ndarray:
        scores = np.asarray(self.pipeline.predict(frame), dtype=float)
        return np.clip(scores, 1e-6, None)

    def predict_proba(self, request: PredictSurveyRequest) -> list[float]:
        frame = build_request_frame(request)
        scores = self.predict_scores(frame)
        return normalize_distribution(scores).tolist()

    def predict_records(self, records: list[SurveyRecord]) -> dict[str, list[float]]:
        frame = build_option_rows(records)
        scores = self.predict_scores(frame.drop(columns=["target_probability"]))
        frame = frame.copy()
        frame["score"] = scores
        predictions: dict[str, list[float]] = {}
        for record_id, group in frame.groupby("record_id", sort=False):
            ordered = group.sort_values("option_index")
            predictions[str(record_id)] = normalize_distribution(
                ordered["score"].to_numpy(dtype=float)
            ).tolist()
        return predictions

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> SimpleSupervisedSurveyPrior:
        return joblib.load(path)
