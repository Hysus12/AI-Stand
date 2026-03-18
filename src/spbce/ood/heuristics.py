from __future__ import annotations

from dataclasses import dataclass

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from spbce.schema.api import PredictSurveyRequest
from spbce.schema.canonical import SurveyRecord


@dataclass(slots=True)
class OodAssessment:
    score: float
    ood_flag: bool
    notes: list[str]


class TfidfOodDetector:
    def __init__(self, threshold: float = 0.55) -> None:
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
        self.matrix = None
        self.population_signatures: set[str] = set()
        self.domains: set[str] = set()

    def fit(self, records: list[SurveyRecord]) -> TfidfOodDetector:
        corpus = [record.question_text for record in records]
        self.matrix = self.vectorizer.fit_transform(corpus)
        self.population_signatures = {record.population_signature() for record in records}
        self.domains = {record.domain or "unknown" for record in records}
        return self

    def assess(self, request: PredictSurveyRequest) -> OodAssessment:
        if self.matrix is None:
            return OodAssessment(score=1.0, ood_flag=True, notes=["OOD detector is not fitted."])
        vector = self.vectorizer.transform([request.question_text])
        max_similarity = float(cosine_similarity(vector, self.matrix).max())
        score = 1.0 - max_similarity
        notes: list[str] = []
        if score >= self.threshold:
            notes.append("Question wording is weakly supported by training data.")
        population_signature = request.population_struct.signature() or request.population_text
        if population_signature not in self.population_signatures:
            score = max(score, 0.75)
            notes.append("Population signature is not covered in training data.")
        domain = request.context.product_category or "unknown"
        if domain not in self.domains:
            score = max(score, 0.7)
            notes.append("Requested domain is not present in training support.")
        if request.context.price is None and request.context.channel is None:
            score = min(1.0, score + 0.05)
            notes.append("Context is sparse; uncertainty increased.")
        return OodAssessment(score=score, ood_flag=score >= self.threshold, notes=notes)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> TfidfOodDetector:
        return joblib.load(path)
