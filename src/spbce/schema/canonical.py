from __future__ import annotations

from hashlib import sha1
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


def stable_hash(*parts: str) -> str:
    payload = "||".join(part.strip() for part in parts if part).encode("utf-8")
    return sha1(payload, usedforsecurity=False).hexdigest()


class PopulationStruct(BaseModel):
    age_band: str | None = None
    gender: str | None = None
    education: str | None = None
    occupation: str | None = None
    income_band: str | None = None
    region: str | None = None
    ideology: str | None = None
    other: dict[str, Any] = Field(default_factory=dict)

    def signature(self) -> str:
        parts = [
            self.age_band or "",
            self.gender or "",
            self.education or "",
            self.occupation or "",
            self.income_band or "",
            self.region or "",
            self.ideology or "",
        ]
        other_parts = [f"{key}={value}" for key, value in sorted(self.other.items())]
        return "|".join(parts + other_parts)


class SurveyRecord(BaseModel):
    record_id: str
    dataset_id: str
    study_id: str
    group_id: str
    wave_id: str | None = None
    time_start: str | None = None
    time_end: str | None = None
    domain: str | None = None
    country: str | None = None
    language: str | None = None
    population_text: str
    population_struct: PopulationStruct = Field(default_factory=PopulationStruct)
    question_id: str
    question_text: str
    question_topic: str | None = None
    question_type: str
    options: list[str]
    option_order: list[int]
    observed_distribution: list[float]
    sample_size: int | None = None
    weights_available: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("options")
    @classmethod
    def validate_options(cls, value: list[str]) -> list[str]:
        if len(value) < 2:
            raise ValueError("at least two answer options are required")
        return value

    @field_validator("observed_distribution")
    @classmethod
    def validate_distribution(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("observed_distribution cannot be empty")
        if any(probability < 0 for probability in value):
            raise ValueError("observed_distribution cannot contain negative values")
        total = sum(value)
        if total <= 0:
            raise ValueError("observed_distribution must sum to a positive value")
        return [probability / total for probability in value]

    @model_validator(mode="after")
    def validate_shape(self) -> SurveyRecord:
        if len(self.options) != len(self.observed_distribution):
            raise ValueError("options and observed_distribution lengths must match")
        if len(self.option_order) != len(self.options):
            raise ValueError("option_order length must match options length")
        return self

    def population_signature(self) -> str:
        if self.country:
            return f"country={self.country}|{self.population_struct.signature()}"
        return self.population_struct.signature()


class ContextFeatures(BaseModel):
    price: float | None = None
    discount: float | None = None
    channel: str | None = None
    exposure: float | None = None
    inventory_constraint: float | None = None
    capacity_constraint: float | None = None
    seasonality: str | None = None
    campaign_type: str | None = None
    brand_name: str | None = None
    brand_strength: float | None = None
    brand_metadata: dict[str, Any] = Field(default_factory=dict)
    other: dict[str, Any] = Field(default_factory=dict)


class BehaviorSurveyQuestion(BaseModel):
    question_id: str
    question_text: str
    question_topic: str | None = None
    question_type: str
    options: list[str]
    option_order: list[int]
    human_distribution: list[float]
    sample_size: int | None = None
    weights_available: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("human_distribution")
    @classmethod
    def validate_human_distribution(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("human_distribution cannot be empty")
        if any(probability < 0 for probability in value):
            raise ValueError("human_distribution cannot contain negative values")
        total = sum(value)
        if total <= 0:
            raise ValueError("human_distribution must sum to a positive value")
        return [probability / total for probability in value]

    @model_validator(mode="after")
    def validate_question_shape(self) -> BehaviorSurveyQuestion:
        if len(self.options) != len(self.option_order):
            raise ValueError("options and option_order lengths must match")
        if len(self.options) != len(self.human_distribution):
            raise ValueError("options and human_distribution lengths must match")
        return self


class BehaviorOutcome(BaseModel):
    outcome_id: str
    outcome_type: str
    outcome_name: str
    outcome_value: float | dict[str, Any]
    positive_label: str | None = None
    unit: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PairedSurveyBehaviorRecord(BaseModel):
    record_id: str
    dataset_id: str
    study_id: str
    group_id: str
    time_start: str | None = None
    time_end: str | None = None
    domain: str
    population_text: str
    population_struct: PopulationStruct = Field(default_factory=PopulationStruct)
    stimulus_text: str
    questionnaire_id: str | None = None
    survey_questions: list[BehaviorSurveyQuestion]
    survey_distribution_features: dict[str, float] | None = None
    actual_outcome: BehaviorOutcome
    context_features: ContextFeatures = Field(default_factory=ContextFeatures)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def population_signature(self) -> str:
        return self.population_struct.signature()


BehaviorRecord = PairedSurveyBehaviorRecord


def make_survey_record_id(dataset_id: str, question_text: str, population_text: str) -> str:
    return stable_hash(dataset_id, question_text, population_text)
